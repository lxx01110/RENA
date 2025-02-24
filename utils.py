import os
import argparse
import random
import logging
from functools import partial
import numpy as np
import dgl
import torch
import torch.nn as nn
from torch import optim as optim
 

from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.impute import KNNImputer
from scipy.io import loadmat
from torch_geometric.utils import dropout_edge,add_random_edge
from sklearn.model_selection import train_test_split
 
from dgl.data.utils import load_graphs
import dgl
 
from sklearn.preprocessing import StandardScaler


def get_feature_mask(rate, n_nodes, n_features):
    return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()


def ptb_adj(sadj, rate, ptb_type,num_nodes):
    if ptb_type == "add":
        sadj,_= add_random_edge(sadj, p=rate, force_undirected=True, num_nodes=num_nodes)
    elif ptb_type == "remove":
        sadj,_ = dropout_edge(sadj , p=rate, force_undirected=True)
    return sadj

 

def ptb_feature(features,rate,num_nodes,features_dim):
    feature_unmask = get_feature_mask(rate=rate, n_nodes=num_nodes,n_features=features_dim)
    
    features[~feature_unmask] = 0.0
    return features,feature_unmask
 

def convert_csc_to_edge_index(csc_matrix):
    coo_matrix = csc_matrix.tocoo()
    rows = coo_matrix.row
    cols = coo_matrix.col
    edge_array = np.stack((rows, cols), axis=0)
    edge_index = torch.tensor(edge_array, dtype=torch.long)
    return edge_index


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

 
def get_optimized_graph(adj_matrix,edge_index,threshold,num_nodes,label):

    values = torch.ones(edge_index.size(1))
    adj_sparse = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
    adj_dense = adj_sparse.to_dense()
    adj_dense = (adj_dense + adj_dense.t()).clamp(max=1)  #   
 
    adj_matrix = adj_matrix+adj_dense
    adj_matrix [adj_matrix < threshold] = 0
    
    rows, cols = torch.nonzero(adj_matrix, as_tuple=True)
    weights = adj_matrix[rows, cols]
    graph = dgl.graph((rows, cols),num_nodes = num_nodes)
    graph.edata['weight'] = weights
    graph.ndata['label'] = label
    num_classes = torch.max(label)+1

    return graph, num_classes


def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx
 

def get_knn_imputed(features,rate,num_nodes,features_dim):
     
    feature_unmask = get_feature_mask(rate=rate, n_nodes=num_nodes,n_features=features_dim)
    features[~feature_unmask] = float('nan')
    imputer = KNNImputer(n_neighbors=8)
    adj_matrix_filled = imputer.fit_transform(features.numpy())
    adj_matrix_filled = torch.tensor(adj_matrix_filled)
    return adj_matrix_filled,feature_unmask
 

def load_dataset(dataset_name,threshold,ptb_type,seed,r,T,p,ptb_rate):
    if dataset_name in ["amazon", "yelp"]:
        if dataset_name == "amazon":
            data = loadmat('data/Amazon.mat')  
            #adj_matrix = torch.load(f'/lea_str/amazonadd_pos_ration_0.4_neg_times_3_ramdom_times_9_{seed}.pt')
            #adj_matrix = torch.load(f'amazon_add_p_{p}_r_{r}_T_{T}_{seed}.pt', map_location='cpu')                                                                 
        if dataset_name == "yelp":
            data = loadmat('data/YelpChi.mat')
            #adj_matrix = torch.load(f'/storage/lixx/data/adj_learning_yelp/yelp_add_pos_ration_{p}_neg_times_{r}_random_times_{T}__{seed}.pt', map_location='cpu')
        f = data['features']
        l = data['label'].squeeze(0)
        features = sp.csr_matrix(f, dtype=np.float32)
        features = torch.FloatTensor(np.array(features.todense())) 
        label = torch.LongTensor(np.array(l))
        edge_index = convert_csc_to_edge_index(data['homo'])
        edge_index = ptb_adj(edge_index,0.3,ptb_type,features.size(0))
        
    if dataset_name == "tfinance":
        graph, label_dict = load_graphs('data/tfinance')
        graph = graph[0]
        graph.ndata['label'] = graph.ndata['label'].argmax(1)
        label = graph.ndata['label'].long().squeeze(-1)
        features = graph.ndata['feature'].float()
        src, dst=graph.edges()
        edge_index = torch.from_numpy(np.stack((src, dst), axis=0))
        edge_index = ptb_adj(edge_index,0.3,ptb_type,features.size(0))
        #adj_matrix = torch.load(f'/storage/lixx/data/adj_learning_albation_2/tfinanceadd_pos_ration_0.2_neg_times_{r}_ramdom_times_2_{seed}.pt', map_location='cpu')
        #adj_matrix = torch.load(f'/storage/lixx/data/adj_learning_albation/tfinanceadd_pos_ration_{pos_ratio}_neg_times_2_random_times_3__{seed}.pt', map_location='cpu')
    #import pdb;pdb.set_trace()
    adj_matrix = torch.load(f'/storage/lixx/data/rena/{dataset_name}/add_p_{p}_r_{r}_T_{T}_{seed}.pt', map_location='cpu')
    #adj_matrix = torch.load(f'/storage/lixx/data/adj_learning_albation/{dataset_name}add_pos_ration_{pos_ratio}_neg_times_{neg_times}_random_times_{random_times}_{seed}.pt', map_location='cpu')
    graph,num_classes  =get_optimized_graph(adj_matrix,edge_index,threshold,features.size(0),label)
     
    if dataset_name == 'amazon':
        index = np.array(list(range(3305, len(graph.ndata['label']))))
    else:
        index = np.arange(len(graph.ndata['label']))
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, graph.ndata['label'][index], stratify=graph.ndata['label'][index],
                                                            train_size=0.4,
                                                            random_state=2, shuffle=True)
    idx_val, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    features,features_unmask = get_knn_imputed(features,ptb_rate,features.size(0),features.size(1))
    num_features = features.shape[1]
    num_nodes = features.shape[0]
   
    features = scale_feats(features)

    graph.ndata['feat'] = features
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    train_mask = torch.full((num_nodes,), False).index_fill_(0, torch.from_numpy(idx_train), True)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, torch.from_numpy(idx_val), True)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, torch.from_numpy(idx_test), True)
    
    graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    
    train_mask = torch.zeros([len(graph.ndata['label'])]).bool()
    train_mask[idx_train] = 1
    weight = (1-graph.ndata['label'][train_mask]).sum().item() / graph.ndata['label'][train_mask].sum().item()
    return graph, (num_features, num_classes),features_unmask,weight




def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def T_f1(idx_val,idx_test,probs,label):
     
    f1, thres = get_best_f1(label[idx_val], probs[idx_val])
    preds = np.zeros_like(label)
    preds[probs[:, 1] > thres] = 1
    trec = recall_score(label[idx_test], preds[idx_test])
    tpre = precision_score(label[idx_test], preds[idx_test])
    tmf1 = f1_score(label[idx_test], preds[idx_test], average='macro')
    tauc = roc_auc_score(label[idx_test], probs[idx_test][:, 1])

    confusion_m = metrics.confusion_matrix(label[idx_test], torch.argmax(torch.from_numpy(probs[idx_test]), dim=-1))
     
    tn, fp, fn, tp = confusion_m.ravel()
    gmean = (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5
    return f1,trec,tpre,tmf1,tauc,gmean


 

def set_random_seed(seed, cuda=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
       
    torch.use_deterministic_algorithms(True)
    if cuda is True:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

 
 

def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1])
    parser.add_argument("--dataset", type=str, default="cora")

   
    parser.add_argument("--max_epoch", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=16,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
  
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)

    parser.add_argument("--encoder", type=str, default="gat")
    
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2)
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--max_epoch_f", type=int, default=500)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob",  default=False)
    

    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--concat_hidden", action="store_true", default=False)


    parser.add_argument("--ptb_type", type=str, default="add")
    parser.add_argument("--threshold", type=float, default=0.3, help="learning rate for evaluation")
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--r", type=int, default=1,help="ratio of unconnected node pairs to connected node pairs")
    parser.add_argument("--T", type=int, default=1.0, help="number of structure view")
    parser.add_argument("--p", type=float, default=0.4, help="number of sampled connected node pairs")
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--observed", type=float, default=1.0)
    parser.add_argument("--imputed", type=float, default=0.5)
    args = parser.parse_args()
    
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


 

 

class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
