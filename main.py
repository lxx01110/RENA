import logging
import numpy as np
from tqdm import tqdm
import torch
from utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    load_dataset
)
from evaluation import node_classification_evaluation
from models import build_model
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device):
    logging.info("start training..")
    graph = graph.to(device)
    
    x = feat.to(device)
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(graph, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
    return model


def main(args):
     
    device = torch.device(args.device)
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch_f = args.max_epoch_f
    optim_type = args.optimizer 

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model

    f1_list = []
    auc_list = []
    gmeans_list = []

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(i)                                                        
        graph, (num_features, num_classes),features_unmask,weights = load_dataset(dataset_name,args.threshold,args.ptb_type,seed,args.r,args.T,args.p,args.rate)
         
        print(f"num_features: {num_features}, num_classes: {num_classes}")
        weights=torch.tensor([1., weights]).to(device)
        args.num_features = num_features
        features_unmask = features_unmask.to(device)
        model = build_model(args,features_unmask)
       
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(model, graph, x, optimizer, args.max_epoch, device)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()
        f1,auc,gmeans = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,weights, dataset_type="fraud",linear_prob = linear_prob)
        f1_list.append(f1)
        auc_list.append(auc)
        gmeans_list.append(gmeans)
    
    
    final_f1, final_f1_std = np.mean(f1_list), np.std(f1_list)
    final_auc, final_auc_std = np.mean(auc_list), np.std(auc_list)
    final_gmeans, final_gmeans_std = np.mean(gmeans_list), np.std(gmeans_list)
    
    with open(f'{args.dataset}_result.txt', 'a') as f:
        f.write("\n\n") 
        f.write(f"# python main.py --rate {args.rate} --p {args.p} --observed {args.observed} --imputed {args.imputed} --ptb_type {args.ptb_type} --num_heads {args.num_heads} --threshold {args.threshold} --num_hidden {args.num_hidden} --r {args.r} --T {args.T} --device {args.device} --dataset {args.dataset} --loss_fn {args.loss_fn} \n")
        f.write(f"# final_f1: {final_f1:.4f}±{final_f1_std:.4f}\n")
        f.write(f"# final_auc: {final_auc:.4f}±{final_auc_std:.4f}\n")
        f.write(f"# final_gmeans: {final_gmeans:.4f}±{final_gmeans_std:.4f}\n")
        f.write('\nAll Done!\n')
        f.write('-------------------------------------------\n')
    print(f"# final_f1: {final_f1:.4f}±{final_f1_std:.4f}")
    print(f"# final_auc: {final_auc:.4f}±{final_auc_std:.4f}")
    print(f"# final_gmeans: {final_gmeans:.4f}±{final_gmeans_std:.4f}")

  
 
if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)
