import torch
import dgl
import os
import torch as th
import dgl.function as fn
from cpu_mem_usage import get_memory
import time
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]


def convert_mag_to_homograph(g):
    """
    Featurize node types that don't have input features (i.e. author,
    institution, field_of_study) by averaging their neighbor features.
    Then convert the graph to a undirected homogeneous graph.
    """
    src_writes, dst_writes = g.all_edges(etype="writes")
    src_topic, dst_topic = g.all_edges(etype="has_topic")
    src_aff, dst_aff = g.all_edges(etype="affiliated_with")
    new_g = dgl.heterograph({
        ("paper", "written", "author"): (dst_writes, src_writes),
        ("paper", "has_topic", "field"): (src_topic, dst_topic),
        ("author", "aff", "inst"): (src_aff, dst_aff)
    })
    # new_g = new_g.to(device)
    new_g.nodes["paper"].data["feat"] = g.nodes["paper"].data["feat"]

    new_g["written"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    new_g["has_topic"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    new_g["aff"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))

    g.nodes["author"].data["feat"] = new_g.nodes["author"].data["feat"]
    g.nodes["institution"].data["feat"] = new_g.nodes["inst"].data["feat"]
    g.nodes["field_of_study"].data["feat"] = new_g.nodes["field"].data["feat"]


    # Convert to homogeneous graph
    # Get DGL type id for paper type
    target_type_id = g.get_ntype_id("paper")
    print('target_type_id ',target_type_id)
    g = dgl.to_homogeneous(g, ndata=["feat"])
    g = dgl.add_reverse_edges(g, copy_ndata=True)
    # Mask for paper nodes
    g.ndata["target_mask"] = g.ndata[dgl.NTYPE] == target_type_id
    output, counts = th.unique_consecutive(g.ndata[dgl.NTYPE], return_counts=True)
    print('counts',counts)
    
    return g

def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.R + 1):
        g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
    res = []
    for hop in range(args.R + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))

    if args.dataset == "ogbn-mag":
        # For MAG dataset, only return features for target node types (i.e.
        # paper nodes)
        target_mask = g.ndata["target_mask"]
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_res = []
        for x in res:
            feat = torch.zeros((num_target,) + x.shape[1:],
                               dtype=x.dtype, device=x.device)
            feat[target_ids] = x[target_mask]
            new_res.append(feat)
        res = new_res
    return res

    
def prepare_data(g, n_classes, args, device):

    tmp = (g.in_degrees()==0) & (g.out_degrees()==0)
    isolated_nodes = torch.squeeze(torch.nonzero(tmp, as_tuple=False))
    g.remove_nodes(isolated_nodes)
    
    feats = g.ndata.pop('feat')      
    labels = g.ndata.pop('label')

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]
    print('success----------------------------------------')
    print(len(train_nid))
    print(len(val_nid))
    print(len(test_nid))
    # g.ndata['features'] = g.ndata['feat']
    # g.ndata['labels'] = g.ndata['label']
    # feat = g.ndata.pop('feat')
    # label = g.ndata.pop('label')
    
    data = g,  feats, labels, n_classes, train_nid, val_nid, test_nid
    return data
    


def load_ogbn_mag(args):
    dataset_name = args.dataset
    dataset = DglNodePropPredDataset(name=dataset_name)
    raw_g, labels = dataset[0]
    homo_g = convert_mag_to_homograph(raw_g)
    paper_labels = labels['paper'].squeeze()

    split_idx = dataset.get_idx_split()
    train_nid = split_idx["train"]['paper']
    val_nid = split_idx["valid"]['paper']
    test_nid = split_idx["test"]['paper']
   
    n_classes = dataset.num_classes
    
    print(f"# total Nodes: {homo_g.number_of_nodes()}\n"
          f"# total Edges: {homo_g.number_of_edges()}\n"
          f"# paper graph Labels: {len(paper_labels)}\n"
          f"# paper graph Train: {len(train_nid)}\n"
          f"# paper graph Val: {len(val_nid)}\n"
          f"# paper graph Test: {len(test_nid)}\n"
          f"# paper graph Classes: {n_classes}")

    feats = neighbor_average_features(homo_g, args)
    
    feats = feats[0]# we only keep the 1-hop neighbor mean feature value
    g = dgl.node_subgraph(homo_g,homo_g.ndata["target_mask"])
    # g.ndata['features']= feats
    # g.ndata['labels']=paper_labels
    g.ndata.pop('feat') 
    # g.ndata.pop('features') 
    
    return g, feats, paper_labels, n_classes, train_nid, val_nid, test_nid
   

def preprocess_papers100M(args):
    dataset = DglNodePropPredDataset(name=args.dataset)
    g, labels = dataset[0]     
    print('--------------------------------------preprocess the papers100M graph')
    srcs, dsts = g.all_edges()
    g.add_edges(dsts, srcs)
    labels = labels.view(-1).type(torch.int)
    splitted_idx = dataset.get_idx_split()
    
    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    name = args.dataset
    print(name)

    n_classes = dataset.num_classes        
    labels = labels.squeeze()
    evaluator = get_ogb_evaluator(name)        
    print(f"# Nodes: {g.number_of_nodes()}\n"
        f"# Edges: {g.number_of_edges()}\n"
        f"# Train: {len(train_nid)}\n"
        f"# Val: {len(val_nid)}\n"
        f"# Test: {len(test_nid)}\n"
        f"# Classes: {n_classes}\n")

    in_feats = g.ndata['feat'].shape[1]
    train_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    tmp = (g.in_degrees()==0) & (g.out_degrees()==0)
    isolated_nodes = torch.squeeze(torch.nonzero(tmp, as_tuple=False))
    g.remove_nodes(isolated_nodes)
    import os
    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    print(str(tot_m)+ ' '+ str(used_m) + ' '+ str(free_m))

    save_graphs('./DATA/'+args.dataset+'_homo_without_isolated_node_graph.bin',[g])
    print('--------------------------------------save the papers100M graph to DATA folder')



def prepare_data_papers100m(device, args):
    dataset = DglNodePropPredDataset(name=args.dataset)
    g, labels = dataset[0]     
    print('--------------------------------------print the papers100M graph')
    
    srcs, dsts = g.all_edges()
    g.add_edges(dsts, srcs)
    labels = labels.view(-1).type(torch.int)
    print('labels')
    print(len(labels))
    splitted_idx = dataset.get_idx_split()
    
    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    print('len(train_nid) len(val_nid) len(test_nid)')
    print(len(train_nid))
    print(len(val_nid))
    print(len(test_nid))
    print(get_memory('----------------------------------------print(len train nid)'))
    

    name = args.dataset
    print(name)

    n_classes = dataset.num_classes        
    labels = labels.squeeze()
    # evaluator = get_ogb_evaluator(name)        
    print(f"# Nodes: {g.number_of_nodes()}\n"
        f"# Edges: {g.number_of_edges()}\n"
        f"# Train: {len(train_nid)}\n"
        f"# Val: {len(val_nid)}\n"
        f"# Test: {len(test_nid)}\n"
        f"# Classes: {n_classes}\n")
    print('----------------------------------original graph')
    print(g)
    print(g.ndata)

    nfeat = g.ndata.pop('feat')
    in_feats = nfeat.shape[1]
    print(get_memory('----------------------------------------print(in_feats = nfeat.shape[1])'))
    # feats = neighbor_average_features(total_g, args)

    # labels = labels.to(device)
    # train_nid = train_nid.to(device)
    # val_nid = val_nid.to(device)
    # test_nid = test_nid.to(device)
    
    
    # g.ndata['labels']=labels

    # sub_graph = dgl.node_subgraph(total_g,total_g.ndata["target_mask"])
    # sub_graph.ndata['labels'] = sub_graph.ndata['label']=labels
    # train_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    # train_mask[train_nid] = True
    # val_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    # val_mask[val_nid] = True
    # test_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    # test_mask[test_nid] = True
    # g.ndata['train_mask'] = train_mask
    # g.ndata['val_mask'] = val_mask
    # g.ndata['test_mask'] = test_mask
    print(get_memory('----------------------------------------print(g.ndata["test_mask"] = test_mask'))
    

    # tmp = (g.in_degrees()==0) & (g.out_degrees()==0)
    # isolated_nodes = torch.squeeze(torch.nonzero(tmp, as_tuple=False))
    # g.remove_nodes(isolated_nodes)
    # print(get_memory('----------------------------------------print(after removing isolated nodes'))
    print('after removing isolated nodes')
    print(g.ndata)
    # save_graphs('./DATA/'+args.dataset+'_homo_without_isolated_node_graph.bin',[g])
    # print('write bin success')


    # train_g = val_g = test_g = g
    # train_g = g
    # train_labels = val_labels = test_labels = labels
    # train_labels = labels
    # train_nfeat = g.ndata['feat']
    # train_nfeat = val_nfeat = test_nfeat = g.ndata['feat']
    # train_g.create_formats_()
    # val_g.create_formats_()
    # test_g.create_formats_()

    # return
    return n_classes, nfeat, in_feats, g, labels, train_nid, val_nid, test_nid

   

def ttt(tic, str1):
    toc = time.time()
    print(str1 + '\n step Time(s): {:.4f}'.format(toc - tic))
    return toc


def load_ogbn_dataset(args):
    """
    Load dataset and move graph and features
    """
    '''if name not in ["ogbn-products", "ogbn-arxiv","ogbn-mag"]:
        raise RuntimeError("Dataset {} is not supported".format(name))'''
    name=args.dataset
    if name not in ["ogbn-products", "ogbn-mag","ogbn-papers100M"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name, root=os.path.join(os.environ['HOME'], 'data', 'OGB'))
    splitted_idx = dataset.get_idx_split()
    print(name)

    if name=="ogbn-papers100M":
        
    
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]  
        labels = labels.view(-1).type(torch.int)
        
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)  

        n_classes = dataset.num_classes        
        # labels = labels.squeeze()

        # evaluator = get_ogb_evaluator(name)        
        print(f"# Nodes: {g.number_of_nodes()}\n"
            f"# Edges: {g.number_of_edges()}\n"
            f"# Train: {len(train_nid)}\n"
            f"# Val: {len(val_nid)}\n"
            f"# Test: {len(test_nid)}\n"
            f"# Classes: {n_classes}\n")

        return g, labels, n_classes, train_nid, val_nid, test_nid




def load_karate():
    from dgl.data import KarateClubDataset

    # load Karate data
    data = KarateClubDataset()
    g = data[0]
    print('karate data')
    # print(data[0].ndata)
    # print(data[0].edata)
    ndata=[]
    for nid in range(34):
        ndata.append((th.ones(4)*nid).tolist())
    ddd = {'feat': th.tensor(ndata)}
   
    g.ndata['feat'] = ddd['feat']
    # print(data[0].ndata)
    # g.ndata['labels'] = g.ndata['label']

    train = [True]*24 + [False]*10
    val = [False] * 24 + [True] * 5 + [False] * 5
    test = [False] * 24 + [False] * 5 + [True] * 5
    g.ndata['train_mask'] = th.tensor(train)
    g.ndata['val_mask'] = th.tensor(val)
    g.ndata['test_mask'] = th.tensor(test)

    return g, data.num_classes


def load_cora():
    from dgl.data import CoraGraphDataset
    # load cora data
    data = CoraGraphDataset()
    g = data[0]
    # g.ndata['features'] = g.ndata['feat']
    # g.ndata['labels'] = g.ndata['label']
    return g, data.num_classes


def load_reddit():
    from dgl.data import RedditDataset
    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g = dgl.remove_self_loop(g)
    return g, data.num_classes

def load_ogb(name):

    data = DglNodePropPredDataset(name=name,root=os.path.join(os.environ['HOME'], 'data', 'OGB'))
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]

    graph = dgl.remove_self_loop(graph) 

    labels = labels[:, 0]
    graph.ndata['label'] = labels
   
    in_feats = graph.ndata['feat'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    
    return graph, num_labels

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g
