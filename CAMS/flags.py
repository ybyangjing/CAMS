import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', default='configs/utzappos.yml', help='Path of the config file') 
parser.add_argument('--dataset_path', default='ut-zap50k', help='Path of the dataset')
parser.add_argument('--splitname', default='compositional-split-natural', help="Dataset split")
parser.add_argument('--open_world', action='store_true', default=False, help='perform open world experiment')
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument('--bias', type=float, default=1e3, help='Bias value for unseen concepts')
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-5) #1e-5
parser.add_argument("--save_every_n", default=5, type=int, help="saves the model every n epochs")

parser.add_argument('--topk', type=int, default=1,help="Compute topk accuracy")
parser.add_argument('--threshold', default=None,help="Apply a specific threshold at test time for the hard masking")
parser.add_argument('--threshold_trials', type=int, default=50,help="how many threshold values to try")