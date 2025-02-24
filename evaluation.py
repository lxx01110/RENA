
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import create_optimizer, T_f1


def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,weights, dataset_type,linear_prob=True, mute=False):
     
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    best_f1,best_auc,best_gmeans  = linear_probing_for_fraud_node_classiifcation(encoder, graph, x, optimizer_f, max_epoch_f, device,weights, mute)
    return best_f1,best_auc,best_gmeans
   
   

def linear_probing_for_fraud_node_classiifcation(model, graph, feat, optimizer, max_epoch, device,weights, mute=False):
    
    criterion = F.cross_entropy
    x = feat.to(device)
    graph = graph.to(device)
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]
    
    best_f1 ,best_tauc,best_tmf1, best_gmeans= 0,0,0,0
    best_val ,best_test,best_train= 0,0,0
    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask],weight = weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
        if weights.size(0)==2:
            f1,trec,tpre,tmf1,tauc,gmean= T_f1(val_mask.cpu().numpy(),test_mask.cpu().numpy(),pred.detach().cpu().numpy(),labels.detach().cpu().numpy())
           
            if best_f1 < f1:
                best_f1 = f1
                best_tmf1 = tmf1
                best_tauc = tauc
                best_gmeans = gmean
            if not mute:
                epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val mf1: {tmf1:.4f}, (best f1 : {best_f1:.4f}),auc : {best_tauc:.4f} , gmeans : {best_gmeans:.4f}")  
    return best_tmf1,best_tauc,best_gmeans


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits


 