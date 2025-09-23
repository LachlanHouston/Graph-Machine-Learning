import numpy as np
import random
import scipy.sparse as sp
import torch
from texttable import Texttable
from collections import defaultdict


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def args_print(args):
    _dict = vars(args)
    t = Texttable() 
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def randint():
    return np.random.randint(2**32 - 1)



def feature_OAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        
        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),\
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)
        
        times[_type]   = tims
        indxs[_type]   = idxs
        
        if _type == 'paper':
            texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
    return feature, times, indxs, texts


def feature_MAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()), dtype = int)
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        feature[_type] = graph.node_feature[_type][idxs]
        times[_type]   = tims
        indxs[_type]   = idxs
        
    return feature, times, indxs, texts

def sample_subgraph(graph, sampled_depth=2, sampled_number=8, inp=None,
                    feature_extractor=feature_OAG, enforce_causality=True,
                    relation_blocklist={'has_category', 'rev_has_category'}):
    relation_blocklist = set(relation_blocklist or [])
    '''
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, time>.
        Currently sampled nodes are stored in layer_data.
        After nodes are sampled, we construct the sampled adjacancy matrix.
    '''
    layer_data  = defaultdict( #target_type
                        lambda: {} # {target_id: [ser, time]}
                    )
    budget     = defaultdict( #source_type
                                    lambda: defaultdict(  #source_id
                                        lambda: [0., 0] #[sampled_score, time]
                            ))
    new_layer_adj  = defaultdict( #target_type
                                    lambda: defaultdict(  #source_type
                                        lambda: defaultdict(  #relation_type
                                            lambda: [] #[target_id, source_id]
                                )))
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    '''
    def add_budget(te, target_id, target_time, layer_data, budget):
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                # skip blocked relations
                if relation_type in relation_blocklist:
                    continue
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
                for source_id in sampled_ids:
                    e = adl[source_id]
                    source_time = target_time if (e is None) else e
                    if source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += 1. / len(sampled_ids)
                    budget[source_type][source_id][1] = source_time


    '''
        First adding the sampled nodes then updating budget.
    '''
    for _type in inp:
        for _id, _time in inp[_type]:
            layer_data[_type][_id] = [len(layer_data[_type]), _time]
    for _type in inp:
        te = graph.edge_list[_type]
        for _id, _time in inp[_type]:
            add_budget(te, _id, _time, layer_data, budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.edge_list[source_type]
            keys  = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                '''
                    Directly sample all the nodes
                '''
                sampled_ids = np.arange(len(keys))
            else:
                '''
                    Sample based on accumulated degree
                '''
                score = np.array(list(budget[source_type].values()))[:,0] ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False) 
            sampled_keys = keys[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1]]
            for k in sampled_keys:
                add_budget(te, k, budget[source_type][k][1], layer_data, budget)
                budget[source_type].pop(k)   
    '''
        Prepare feature, time and adjacency matrix for the sampled graph
    '''
    feature, times, indxs, texts = feature_extractor(layer_data, graph)
            
    edge_list = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    edge_time_list = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # self loops unchanged
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            edge_list[_type][_type]['self'].append([_ser, _ser])
            edge_time_list[_type][_type]['self'].append(0)  # or a neutral bucket

    # rebuild edges from original graph
    for target_type in graph.edge_list:
        tld = layer_data.get(target_type, {})
        if not tld: 
            continue
        te = graph.edge_list[target_type]
        for source_type in te:
            sld = layer_data.get(source_type, {})
            if not sld:
                continue
            tes = te[source_type]
            for relation_type, tesr in tes.items():
                if relation_type in relation_blocklist:
                    continue
                for target_key, src_dict in tld.items():
                    if target_key not in tesr:
                        continue
                    target_ser = src_dict[0]
                    t_target = src_dict[1]  # target node time
                    for source_key, e_time in tesr[target_key].items():
                        if source_key not in sld:
                            continue
                        if e_time is None:
                            e_time = layer_data[source_type][source_key][1]
                        if enforce_causality and (e_time is not None) and (e_time > t_target):
                            continue
                        source_ser = layer_data[source_type][source_key][0]
                        edge_list[target_type][source_type][relation_type].append([target_ser, source_ser])
                        edge_time_list[target_type][source_type][relation_type].append(e_time if e_time is not None else t_target)
    return feature, times, edge_list, edge_time_list, indxs, texts

def to_torch(feature, time, edge_list, edge_time_list, graph, device,
             relation_blocklist={'has_category','rev_has_category'},
             type_blocklist=None,
             bucket_days=30,
             max_span_days=365*5):
    relation_blocklist = set(relation_blocklist or [])
    type_blocklist = set(type_blocklist or [])

    node_dict = {}
    node_feature, node_type, node_time = [], [], []

    # Build node_dict from sampled types present in 'feature'
    present_types = [t for t in feature.keys() if t not in type_blocklist]
    offset = 0
    for t in present_types:
        node_dict[t] = [offset, len(node_dict)]
        offset += len(feature[t])

    for t in present_types:
        node_feature += list(feature[t])
        node_type    += [node_dict[t][1]] * len(feature[t])
        node_time    += list(time[t])

    # Might want to bucketize in future
    def bucketize(dt_days):
        # clip to window and shift to non-negative
        span = max_span_days
        dt = int(np.clip(dt_days, -span, span))
        half = span // bucket_days
        return (dt // bucket_days) + half  # integer index in [0, 2*half]

    edge_index, edge_type, edge_time = [], [], []

    edge_dict = {}
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)

    # iterate edges
    for tt in edge_list:
        if tt not in node_dict: continue
        t_off = node_dict[tt][0]
        for st in edge_list[tt]:
            if st not in node_dict: continue
            s_off = node_dict[st][0]
            for rel, pairs in edge_list[tt][st].items():
                if rel in relation_blocklist: 
                    continue
                if rel not in edge_dict:
                    continue
                times_for_rel = edge_time_list[tt][st][rel]
                for (ti, si), e_time_abs in zip(pairs, times_for_rel):
                    tid = ti + t_off
                    sid = si + s_off

                    t_abs = node_time[tid]
                    dt = t_abs - e_time_abs
                    edge_index.append([sid, tid])
                    edge_type.append(edge_dict[rel])
                    edge_time.append(dt)

    # Tensorize
    device = torch.device(device)
    node_feature = torch.tensor(node_feature, dtype=torch.float32, device=device)
    node_type    = torch.tensor(node_type,    dtype=torch.long,   device=device)
    if len(edge_index) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long, device=device)
        edge_type  = torch.empty((0,),  dtype=torch.long, device=device)
        edge_time  = torch.empty((0,),  dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        edge_type  = torch.tensor(edge_type,  dtype=torch.long, device=device)
        edge_time  = torch.tensor(edge_time,  dtype=torch.long, device=device)

    return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict

def graph_sample(graph, target, args, seed, samp_nodes, device='cpu'):
    np.random.seed(seed)
    feature, times, edge_list, edge_time_list, indxs, _ = sample_subgraph(
        graph,
        inp={target: np.stack([samp_nodes, graph.years[samp_nodes]], axis=1)},
        sampled_depth=args.sample_depth,
        sampled_number=args.sample_width,
        feature_extractor=feature_MAG,
        relation_blocklist={'has_category','rev_has_category'},
        enforce_causality=False,
    )

    node_feature, node_type, edge_time, edge_index, edge_type, *_ = \
        to_torch(feature, times, edge_list, edge_time_list, graph, device)

    train_mask = graph.train_mask[indxs[target]]
    valid_mask = graph.valid_mask[indxs[target]]
    test_mask  = graph.test_mask[indxs[target]]
    ylabel     = graph.y[indxs[target]]
    return node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, valid_mask, test_mask), ylabel

def prepare_data(pool, graph, target, target_nodes, args, task_type='train', target_type='business', s_idx=0, n_batch=None, batch_size=None, device='cpu'):
    n_batch = n_batch if n_batch is not None else args.n_batch
    batch_size = batch_size if batch_size is not None else args.batch_size

    jobs = []
    if task_type == 'train':
        for _ in range(n_batch):
            samp_nodes = np.random.choice(target_nodes, batch_size, replace=False)
            jobs.append(pool.apply_async(
                graph_sample, args=(graph, target_type, args, random.randint(0, 2**31-1), samp_nodes, device)
            ))
    elif task_type == 'sequential':
        for i in range(n_batch):
            target = graph.test_paper[(s_idx + i) * batch_size : (s_idx + i + 1) * batch_size]
            jobs.append(pool.apply_async(
                graph_sample, args=(graph, target, args, random.randint(0, 2**31 - 1), target, device)
            ))
    elif task_type == 'variance_reduce':
        target = graph.test_paper[s_idx * batch_size : (s_idx + 1) * batch_size]
        for _ in range(n_batch):
            jobs.append(pool.apply_async(
                graph_sample, args=(graph, target, args, random.randint(0, 2**31 - 1), target, device)
            ))
    return jobs

def multilabel_f1_from_logits(logits, targets, threshold=0.5, average="micro", eps=1e-8):
    """
    logits: [B, C], raw
    targets: [B, C], {0,1} floats
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(targets.dtype)

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    if average == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return f1.item()
    elif average == "macro":
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1_c = 2 * precision * recall / (precision + recall + eps)
        return f1_c.mean().item()
    else:
        raise ValueError("average must be 'micro' or 'macro'")