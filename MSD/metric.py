import torch
# loss function for MSV, as Eq.6
def MSV(graph_recon, pos_indices, pos_values, neg_indices, neg_values,BCE):
    pos_losses = []
    for i in range(len(pos_indices)):
        pos_preds_logits = graph_recon[pos_indices[i][0], pos_indices[i][1]]
        pos_loss = torch.mean(BCE(pos_preds_logits, pos_values[i]))
        pos_losses.append(pos_loss)
  
    neg_preds_logits = graph_recon[neg_indices[0], neg_indices[1]]
    neg_loss = torch.mean(BCE(neg_preds_logits, neg_values))

    avg_loss = (pos_loss + neg_loss) / 2
    return avg_loss
