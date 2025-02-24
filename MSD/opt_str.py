import argparse
import tqdm
from dataset import load_dataset
from metric import MSV
from parse import parser_add_main_args
import torch
import numpy as np
import random
from graph_learner import Model
import os
import time
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
 
def fix_seed(seed, cuda=True):
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
### Parse args ###

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

device = args.device

 
edge_index, features, label = load_dataset(args.dataset,args.rate,args.ptb_type)

edge_index = edge_index.to(device)
features = features.to(device)
label = label.to(device)

num_nodes = features.size(0)
num_labels = torch.max(label).item() + 1
num_features = features.size(1)
print(f"num nodes {num_nodes} | num classes {num_labels} | num node feats {num_features}")


edge_index_size = edge_index.size(1)
BCE = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.r]).to(device))
for seed in tqdm(range(5)):                                               
    fix_seed(seed)                                                     
    
    model = Model(args, num_features, features, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_time = time.time()
    epoch = tqdm(range(300))
    min_loss = 100000
    best_adj = None
    early_stop = 100
    cnt = 0
    num_pos_samples = int(edge_index_size*args.p) # number of connected node pairs
    num_neg_samples = int(num_pos_samples*args.r)  # ratio of unconnected node pairs to connected node pairs

    pos_perms=[]
    pos_indices=[]
    pos_values=[]

    neg_indices =[]
    neg_values =[]
    for _ in range(args.T):  # number of structure views
            pos_perm = torch.randperm(edge_index.size(1))[:num_pos_samples]
            pos_indice = edge_index[:, pos_perm]
            pos_value= torch.ones(num_pos_samples).to(device)
            pos_indices.append(pos_indice)
            pos_values.append(pos_value)

    neg_indice = negative_sampling(edge_index=edge_index, num_neg_samples=num_neg_samples)
     
    neg_value = torch.zeros(num_neg_samples).to(device)
    for i in epoch:
        model.train()
        optimizer.zero_grad()
        adj = model()
        loss = MSV(adj,pos_indices,pos_values,neg_indice,neg_value,BCE)
        epoch.set_description("Epoch: {:03d}, Loss: {:.4f}".format(i, loss.item()))
        
        loss.backward()
        optimizer.step()
        if loss.item() < min_loss:
            min_loss = loss.item()
            best_adj = adj.detach().cpu()
            cnt = 0
        else:
            cnt+=1
        if cnt == early_stop:
            break
    # 判断文件夹是否存在
    if not os.path.exists(f"data/rena/{args.dataset}"):
        os.makedirs(f"data/rena/{args.dataset}")
    torch.save(best_adj, f"data/rena/{args.dataset}/{args.ptb_type}_p_{args.p}_r_{args.r}_T_{args.T}_{seed}.pt")
    
    end_time = time.time()
    print("="*20+f'Running time: {end_time-start_time:.2f}s'+"="*20)
    
    