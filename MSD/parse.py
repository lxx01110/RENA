def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--ptb_type', type=str, default='add',help='add or remove edges')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rate', type=float, default=0.3, help='perturbation rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
     
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--p', type=float, default=0.2)
    
    parser.add_argument('--graph_learn_num_pers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--r', type=int, default=1, help='ratio of unconnected node pairs to connected node pairs')
    parser.add_argument('--T', type=int, default=1, help='number of structure views')
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
   