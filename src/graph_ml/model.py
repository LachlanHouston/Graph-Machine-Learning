import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax as pyg_softmax
import math

def softmax_mps_safe(src, index, ptr=None, num_nodes=None):
    if src.device.type != "mps":
        return pyg_softmax(src, index, num_nodes=num_nodes)
    
    # MPS path: do it on CPU to avoid scatter_reduce
    out_cpu = pyg_softmax(src.float().cpu(), index.cpu(), num_nodes=num_nodes)
    return out_cpu.to(device=src.device, dtype=src.dtype)

class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None

        # Attention mechanism layers
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()

        # Target-Specific Learnable Layer
        self.a_linears = nn.ModuleList()

        # Norms
        self.norms = nn.ModuleList()

        for t in range(num_types): # num_types is number of node types
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))

            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        
        self.relation_prior = nn.Parameter(torch.ones(num_relations, self.n_heads)) # Mu prior for edge scaled Softmax
        self.relation_attention = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k)) # Learnable prior for Q and K keys
        self.relation_message = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k)) # Learnable prior for V keys

        self.skip = nn.Parameter(torch.ones(num_types)) # Learnable skip connection
        self.drop = nn.Dropout(dropout)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        """
        Index j is the source node, i is the target node. Performs the Mutual Attention and Message Passing of the model.
        
        """
        data_size = edge_index_i.size(0)
        att_tensor = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        msg_tensor = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type)) # might be sb = source_batch
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]

            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]

                for rel_type in range(self.num_relations):
                    idx = (edge_type == int(rel_type)) & tb

                    if idx.sum() == 0:
                        continue

                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]

                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.relation_attention[rel_type]).transpose(1, 0)
                    att_tensor[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_prior[rel_type] / self.sqrt_dk

                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    msg_tensor[idx] = torch.bmm(v_mat.transpose(1, 0), self.relation_message[rel_type]).transpose(1, 0)

        self.att = softmax_mps_safe(att_tensor, edge_index_i)
        out = msg_tensor * self.att.view(-1, self.n_heads, 1)    
        return out.view(-1, self.out_dim)
    
    def update(self, aggr_out, node_inp, node_type):
        """
        Performs the Target-Specific Aggregation step.
        
        """

        aggr_out = F.gelu(aggr_out)
        out = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)

        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue

            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            alpha = torch.sigmoid(self.skip[target_type])

            if self.use_norm:
                out[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                out[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha) 
        return out
    
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time = edge_time)
    
class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.2, conv_name = 'hgt', prev_norm = False, last_norm = False, use_RTE = True):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(HGTConv(n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(HGTConv(n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        dtype = node_feature.dtype
        res = torch.zeros(node_feature.size(0), self.n_hid, device=node_feature.device, dtype=dtype)
        #res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs  
    
class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid    = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        x = self.linear(x)
        return x