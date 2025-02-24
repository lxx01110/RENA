import numpy as np
import torch
from dgl.data.utils import load_graphs
import scipy.sparse as sp
import ssl
from torch_geometric.utils import dropout_edge,add_random_edge
from scipy.io import loadmat
ssl._create_default_https_context = ssl._create_unverified_context


def convert_csc_to_edge_index(csc_matrix):
    coo_matrix = csc_matrix.tocoo()
    rows = coo_matrix.row
    cols = coo_matrix.col
    edge_array = np.stack((rows, cols), axis=0)
    edge_index = torch.tensor(edge_array, dtype=torch.long)
    return edge_index

 
def get_feature_mask(rate, n_nodes, n_features):
    return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()

def ptb_adj(sadj, rate, ptb_type,num_nodes):
    if ptb_type == "add":
        sadj,_= add_random_edge(sadj, p=rate, force_undirected=True, num_nodes=num_nodes)
    elif ptb_type == "remove":
        sadj,_ = dropout_edge(sadj , p=rate, force_undirected=True)
    else:
        raise ValueError("Invalid ptb_type")
    return sadj

def ptb_feature(features,rate,num_nodes,features_dim):
    feature_mask = get_feature_mask(rate=rate, n_nodes=num_nodes,n_features=features_dim)
    features[~feature_mask] = 0.0
    return features,feature_mask

 

def load_dataset(filename,rate,ptb_type):
    if filename == 'amazon':
        data = loadmat('/storage/lixx/data/Amazon.mat')
        f = data['features']
        l = data['label'].squeeze(0)
        label = torch.LongTensor(np.array(l))
        features = sp.csr_matrix(f, dtype=np.float32)
        features = torch.FloatTensor(np.array(features.todense())) 
        edge_index = convert_csc_to_edge_index(data['homo'])
    elif filename == 'yelp':
        data = loadmat('/storage/lixx/data/YelpChi.mat')
        f = data['features']
        l = data['label'].squeeze(0)
        label = torch.LongTensor(np.array(l))
        features = sp.csr_matrix(f, dtype=np.float32)
        features = torch.FloatTensor(np.array(features.todense())) 
        edge_index = convert_csc_to_edge_index(data['homo'])
    elif filename == 'tfinance':
        graph, label_dict = load_graphs('/storage/lixx/data/tfinance/tfinance')
        graph = graph[0]
        label = graph.ndata['label'].long().squeeze(-1)
        src, dst=graph.edges()
        edge_index = torch.from_numpy(np.stack((src, dst), axis=0))
        features = graph.ndata['feature'].float()
    features,mask= ptb_feature(features,rate,features.size(0),features.size(1))
    edge_index = ptb_adj(edge_index,rate,ptb_type,features.size(0))
   
    return edge_index, features, label
 
 


 