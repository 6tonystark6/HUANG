import argparse
from multiprocessing import freeze_support

from torch.utils.data import DataLoader

from attacked_methods.DPSH.DPSH import *
from attacked_methods.HashNet.HashNet import *

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
torch.backends.cudnn.enabled = False

class HashingDataset(Dataset):
    def __init__(self,
                 data_path,
                 img_filename,
                 label_filename,
                 transform=transforms.Compose([
                     # transforms.Resize(256),
                     # transforms.CenterCrop(224),
                     transforms.Resize((256, 256)),  # 修改这里
                     transforms.ToTensor()
                 ])):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index]).float()
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


def load_label(filename, data_dir):
    label_filepath = os.path.join(data_dir, filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    return torch.from_numpy(label).float()


import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
# description of data
parser.add_argument('--dataset_name', dest='dataset', default='FLICKR', choices=['FLICKR', 'NUS', 'COCO'],
                    help='name of the dataset')
parser.add_argument('--data_dir', dest='data_dir', default='./data/', help='path of the dataset')
parser.add_argument('--database_file', dest='database_file', default='database_img.txt',
                    help='the image list of database images')
parser.add_argument('--train_file', dest='train_file', default='train_img.txt',
                    help='the image list of training images')
parser.add_argument('--test_file', dest='test_file', default='test_img.txt', help='the image list of test images')
parser.add_argument('--database_label', dest='database_label', default='database_label.txt',
                    help='the label list of database images')
parser.add_argument('--train_label', dest='train_label', default='train_label.txt',
                    help='the label list of training images')
parser.add_argument('--test_label', dest='test_label', default='test_label.txt', help='the label list of test images')
# model
parser.add_argument('--hashing_method', dest='method', default='DPSH', choices=['DPSH', 'HashNet', 'CSQ'],
                    help='deep hashing methods')
parser.add_argument('--backbone', dest='backbone', default='VGG19',
                    choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50'], help='backbone network')
parser.add_argument('--yita', dest='yita', type=int, default=50, help='yita in the dpsh paper')
parser.add_argument('--code_length', dest='bit', type=int, default=32, help='length of the hashing code')
# training or test
parser.add_argument('--train', dest='train', type=bool, default=True, choices=[True, False], help='to train or not')
parser.add_argument('--test', dest='test', type=bool, default=True, choices=[True, False], help='to test or not')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='number of images in one batch')
parser.add_argument('--checkpoint_dir', dest='save', default='attacked_models/', help='models are saved here')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=100, help='number of epoch')
parser.add_argument('--learning_rate', dest='lr', type=float, default=0.05, help='initial learning rate for sgd')
parser.add_argument('--weight_decay', dest='wd', type=float, default=1e-5, help='weight decay for SGD')
args = parser.parse_args()

if __name__ == '__main__':
    freeze_support()

    dset_database = HashingDataset(args.data_dir + args.dataset, args.database_file, args.database_label)
    dset_train = HashingDataset(args.data_dir + args.dataset, args.train_file, args.train_label)
    dset_test = HashingDataset(args.data_dir + args.dataset, args.test_file, args.test_label)
    num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)

    database_loader = DataLoader(dset_database, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    database_labels = load_label(args.database_label, args.data_dir + args.dataset)
    train_labels = load_label(args.train_label, args.data_dir + args.dataset)
    test_labels = load_label(args.test_label, args.data_dir + args.dataset)

    model = None
    if args.method == 'DPSH':
        model = DPSH(args.bit, args.batch_size, args.lr, args.backbone, args.dataset, args.n_epochs, args.wd, args.yita,
                     args.save)
        if args.train:
            model.train_DPSH(train_loader, train_labels, num_train)
        if args.test:
            model.load(str(args.save) + '/DPSH_' + str(args.dataset) + '_' + str(args.bit) + '/DPSH.pth')
            model.test_DPSH(database_loader, test_loader, database_labels, test_labels, num_database, num_test)
    elif args.method == 'HashNet':
        model = HashNet(args.bit, args.batch_size, args.lr, args.backbone, args.dataset, args.n_epochs, args.wd,
                        args.yita,
                        args.save)
        if args.train:
            model.train_hashnet(train_loader, train_labels, num_train)
        if args.test:
            model.load(str(args.save) + '/HashNet_' + str(args.dataset) + '_' + str(args.bit) + '/HashNet.pth')
            model.test_hashnet(database_loader, test_loader, database_labels, test_labels, num_database, num_test)
    # elif args.method == 'CSQ':
    #     model = CSQ(args.bit, args.batch_size, args.lr, args.backbone, args.dataset, args.n_epochs, args.wd, args.yita,
    #                 args.save)
    #     if args.train:
    #         if args.dataset == 'NUS':
    #             num_class = 21
    #             true_hash = 'data/NUS/hash_centers/' + str(args.bit) + '_nus_wide_21_class.pkl'
    #         if args.dataset == 'FLICKR':
    #             num_class = 38
    #             true_hash = 'data/FLICKR/hash_centers/' + str(args.bit) + '_flickr_38_class.pkl'
    #         if args.dataset == 'COCO':
    #             num_class = 80
    #             true_hash = 'data/COCO/hash_centers/' + str(args.bit) + '_coco_80_class.pkl'
    #         Hash_center = torch.load(true_hash)
    #         model.train_CSQ(train_loader, Hash_center, -1)
