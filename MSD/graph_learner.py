import torch
import torch.nn as nn
import torch.nn.functional as F

VERY_SMALL_NUMBER = 1e-12
class GraphLearner(nn.Module):
    def __init__(self, input_size, num_pers=4):
        super(GraphLearner, self).__init__()
        self.weight_tensor = torch.Tensor(num_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def reset_parameters(self):
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, context):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
         
        mask = (attention > 0).detach().float()
        attention = attention * mask + 0 * (1 - mask)
        return attention

class Model(nn.Module):
    """
    model for link prediction only task.
    """

    def __init__(self, args, d,features, device):
        super(Model, self).__init__()
        self.device = device
       # self.graph_skip_conn = args.graph_skip_conn
        self.graph_learner = GraphLearner(input_size=d, num_pers=args.graph_learn_num_pers).to(device)
        self.graph_learner.reset_parameters()
        self.features = features
    def forward(self):
        return self.graph_learner(self.features)
        