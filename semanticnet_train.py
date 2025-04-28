import os
import argparse
import warnings

from attack_model import SemanticModel
from load_data import Dataset_Config, load_dataset, split_dataset, allocate_dataset

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', default='FLICKR', choices=['FLICKR', 'COCO', 'NUS'])
parser.add_argument('--dataset_path', dest='dataset_path', default='./Datasets/', help='path of the dataset')
parser.add_argument('--attacked_method', dest='attacked_method', default='DPSH', choices=['DPSH','HashNet', 'CSQ'])
parser.add_argument('--attacked_models_path', dest='attacked_models_path', default='attacked_models/')
parser.add_argument('--bit', dest='bit', type=int, default=32, choices=[16, 32, 48, 64])
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='number of images in one batch')
parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=10, help='print the debug information every print_freq iterations')
parser.add_argument('--output_path', dest='output_path', default='outputs/', help='models are saved here')
parser.add_argument('--output_dir', dest='output_dir', default='DPSH_FLICKR_32', help='the name of output')
parser.add_argument('--gpu', dest='gpu', type=str, default='0', choices=['0', '1', '2', '3'])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

DataConfigs = Dataset_Config(args.dataset, args.dataset_path)
X, Y, L = load_dataset(DataConfigs.data_path)
X_s, Y_s, L_s = split_dataset(X, Y, L, DataConfigs.query_size, DataConfigs.training_size, DataConfigs.database_size)
image_train, _, label_train, image_database, _, label_database, image_test, _, label_test = allocate_dataset(X_s, Y_s, L_s)

semantic_net = SemanticModel(args=args, DataConfigs=DataConfigs)

semantic_net.train_semanticnet(image_train, label_train)
semantic_net.test_semanticnet(image_test, label_test, image_database, label_database)
