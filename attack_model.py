import os
import numpy as np
import scipy.io as scio
from torchvision import transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb

from model import SemanticNet, GANLoss, get_scheduler
from utils import set_input_images, CalcSim, log_trick, CalcMap, mkdir_p, return_results, calc_hamming
from attacked_model import Attacked_Model


class SemanticModel(nn.Module):
    def __init__(self, args, DataConfigs):
        super(SemanticModel, self).__init__()
        self.bit = args.bit
        self.num_classes = DataConfigs.num_label
        self.dim_image = DataConfigs.tag_dim
        self.batch_size = args.batch_size
        self.model_name = '{}_{}_{}'.format(args.attacked_method, args.dataset, args.bit)
        self.args = args
        self._build_model(args, DataConfigs)
        self._save_setting(args)
        print('self.bit:',self.bit, 'self.num_classes', self.num_classes, 'self.dim_image:', self.dim_image,
              'self.batch_size', self.batch_size, 'self.model_name:', self.model_name, 'self.args:', self.args)

    def _build_model(self, args, Dcfg):
        pretrain_model = scio.loadmat(Dcfg.vgg_path)
        self.semanticnet = nn.DataParallel(SemanticNet(self.dim_image, self.bit, self.num_classes)).cuda()
        self.criterionGAN = GANLoss('lsgan').cuda()
        self.attacked_model = Attacked_Model(args.attacked_method, args.dataset, args.bit, args.attacked_models_path,
                                             args.dataset_path)
        self.attacked_model.eval()

    def _save_setting(self, args):
        self.output_dir = os.path.join(args.output_path, args.output_dir)
        self.model_dir = os.path.join(self.output_dir, 'Model')
        self.image_dir = os.path.join(self.output_dir, 'Image')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)

    def save_semanticnet(self):
        torch.save(self.semanticnet.module.state_dict(),
                   os.path.join(self.model_dir, 'semanticnet_{}.pth'.format(self.model_name)))

    def load_semanticnet(self):
        self.semanticnet.module.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'semanticnet_{}.pth'.format(self.model_name))))
        self.semanticnet.eval()

    def train_semanticnet(self, train_images, train_labels):
        num_train = train_labels.size(0)
        optimizer_a = torch.optim.Adam(self.semanticnet.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        epochs = 100
        batch_size = 64
        steps = num_train // batch_size + 1
        lr_steps = epochs * steps
        scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_a, milestones=[lr_steps / 2, lr_steps * 3 / 4],
                                                           gamma=0.1)
        criterion_l2 = torch.nn.MSELoss()
        # Depends on the attacked model
        B = self.attacked_model.generate_image_hashcode(train_images).cuda()
        for epoch in range(epochs):
            index = np.random.permutation(num_train)
            for i in range(steps):
                end_index = min((i + 1) * batch_size, num_train)
                num_index = end_index - i * batch_size
                ind = index[i * batch_size: end_index]
                batch_image = Variable(train_images[ind]).type(torch.float).cuda()
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                optimizer_a.zero_grad()
                _, mixed_h, mixed_l = self.semanticnet(batch_label, batch_image)
                S = CalcSim(batch_label.cpu(), train_labels.type(torch.float))
                theta_m = mixed_h.mm(Variable(B).t()) / 2
                logloss_m = - ((Variable(S.cuda()) * theta_m - log_trick(theta_m)).sum() / (num_train * num_index))
                regterm_m = (torch.sign(mixed_h) - mixed_h).pow(2).sum() / num_index
                classifer_m = criterion_l2(mixed_l, batch_label)
                loss = classifer_m + 5 * logloss_m + 1e-3 * regterm_m
                loss.backward()
                optimizer_a.step()
                if i % self.args.print_freq == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, l_m:{:.5f}, r_m: {:.5f}, c_m: {:.7f}'
                          .format(epoch + 1, i, scheduler_a.get_last_lr()[0], logloss_m, regterm_m, classifer_m))
                scheduler_a.step()
        self.save_semanticnet()

    def test_semanticnet(self, test_images, test_labels, database_images, database_labels):
        self.load_semanticnet()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        for i in range(num_test):
            _, mixed_h, __ = self.semanticnet(test_labels[i].cuda().float().unsqueeze(0),
                                               test_images[i].cuda().float().unsqueeze(0))
            qB[i, :] = torch.sign(mixed_h.cpu().data)[0]
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        map = CalcMap(qB, IdB, test_labels, database_labels, 50)
        print('MAP: %3.5f' % map)

