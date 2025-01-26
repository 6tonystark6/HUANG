import os
import torch.optim as optim
from torch.autograd import Variable

import math
import torch
import torch.nn as nn
from torchvision import models


class AlexNet(nn.Module):
    def __init__(self, bit):
        super(AlexNet, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = original_model.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = original_model.classifier[1].weight
        cl1.bias = original_model.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[4].weight
        cl2.bias = original_model.classifier[4].bias

        self.classifier = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, bit),
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        # x = (x-self.mean)/self.std
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        y = self.tanh(alpha * y)
        return y


vgg_dict = {"VGG11": models.vgg11, "VGG13": models.vgg13, "VGG16": models.vgg16, "VGG19": models.vgg19,
            "VGG11BN": models.vgg11_bn, "VGG13BN": models.vgg13_bn, "VGG16BN": models.vgg16_bn,
            "VGG19BN": models.vgg19_bn}


class VGG(nn.Module):
    def __init__(self, model_name, bit):
        super(VGG, self).__init__()
        original_model = vgg_dict[model_name](pretrained=True)
        self.features = original_model.features
        self.cl1 = nn.Linear(25088, 4096)
        self.cl1.weight = original_model.classifier[0].weight
        self.cl1.bias = original_model.classifier[0].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[3].weight
        cl2.bias = original_model.classifier[3].bias

        self.classifier = nn.Sequential(
            self.cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, bit),
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        # x = (x-self.mean)/self.std
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        y = self.tanh(alpha * y)
        return y


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, model_name, hash_bit):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[model_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

        self.activation = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        # x = (x-self.mean)/self.std
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        y = self.activation(alpha * y)
        return y


class AlexNetFc(nn.Module):
    def __init__(self, hash_bit):
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        # self.use_hashnet = use_hashnet
        self.hash_layer = nn.Linear(model_alexnet.classifier[6].in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.iter_num = 0
        self.__in_features = hash_bit
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.activation = nn.Tanh()
        self.scale = self.init_scale

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        y = self.hash_layer(x)
        if self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        y = self.activation(self.scale * y)
        return y

    def output_num(self):
        return self.__in_features


class VGGFc(nn.Module):
    def __init__(self, name, hash_bit):
        super(VGGFc, self).__init__()
        model_vgg = vgg_dict[name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        # self.use_hashnet = use_hashnet
        self.hash_layer = nn.Linear(model_vgg.classifier[6].in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.iter_num = 0
        self.__in_features = hash_bit
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.activation = nn.Tanh()
        self.scale = self.init_scale

        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x):
        # x = (x-self.mean)/self.std
        if self.training:
            self.iter_num += 1
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        y = self.hash_layer(x)
        if self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        y = self.activation(self.scale * y)
        return y

    def output_num(self):
        return self.__in_features


class ResNetFc(nn.Module):
    def __init__(self, name, hash_bit):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.iter_num = 0
        self.__in_features = hash_bit
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.activation = nn.Tanh()
        self.scale = self.init_scale

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        if self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        y = self.activation(self.scale * y)
        return y

    def output_num(self):
        return self.__in_features


"""from utils.hamming_matching import *"""

import numpy as np


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcMap(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, int(tsum))

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map


def CalcTopMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap


"""HashNet"""


class HashNet(nn.Module):
    def __init__(self, bit, batch_size, lr, backbone, dataset, n_epochs, wd, yita, save):
        super(HashNet, self).__init__()
        self.alpha = 0.1
        self.bit = bit
        self.batch_size = batch_size
        self.lr = lr
        self.backbone = backbone
        self.n_epochs = n_epochs
        self.wd = wd
        self.yita = yita
        self.save = save
        self.model_name = 'HashNet_{}_{}'.format(dataset, bit)
        print(self.model_name)

        self._build_graph()

    def _build_graph(self):
        if self.backbone == 'AlexNet':
            self.model = AlexNetFc(self.bit)
        elif 'VGG' in self.backbone:
            self.model = VGGFc(self.backbone, self.bit)
        else:
            self.model = ResNetFc(self.backbone, self.bit)
        self.model = self.model.cuda()

    def load(self, path, use_gpu=False):
        if not use_gpu:
            # Load the state_dict
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            # Modify the keys in state_dict to include 'model.' prefix
            new_state_dict = {'model.' + k: v for k, v in state_dict.items()}
            # Load modified state_dict
            self.load_state_dict(new_state_dict)
        else:
            state_dict = torch.load(path)
            new_state_dict = {'model.' + k: v for k, v in state_dict.items()}
            self.load_state_dict(new_state_dict)

    def pairwise_loss_updated(self, u, U, y, Y):
        similarity = (y @ Y.t() > 0).float()
        dot_product = self.alpha * u @ U.t()
        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0
        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S
        return loss

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr * (0.1 ** (epoch // (self.n_epochs // 3)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def generate_whole_code(self, data_loader, num_data):
        B = np.zeros([num_data, self.bit], dtype=np.float32)
        for iter, data in enumerate(data_loader, 0):
            data_input, _, data_ind = data
            data_input = Variable(data_input.cuda())
            output = self.model(data_input)
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        return B

    def generate_code(self, data):
        data_input = Variable(data.cuda())
        output = self.model(data_input)
        return output

    def train_hashnet(self, train_loader, train_labels, num_train):
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr,
                              weight_decay=self.wd)

        U = torch.zeros(num_train, self.bit)

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            # training epoch
            for iter, traindata in enumerate(train_loader, 0):
                train_input, train_label, batch_ind = traindata
                train_label = torch.squeeze(train_label)

                train_input, train_label = Variable(
                    train_input.cuda()), Variable(train_label.cuda())

                self.model.zero_grad()
                train_outputs = self.model(train_input)
                batch_size_ = train_label.size(0)

                for i, ind in enumerate(batch_ind):
                    U[ind, :] = train_outputs.data[i]

                loss = self.pairwise_loss_updated(train_outputs, U.cuda(), train_label, train_labels.cuda())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print('Epoch: %3d/%3d\tTrain_loss: %3.5f' %
                  (epoch + 1, self.n_epochs, epoch_loss / len(train_loader)))
            optimizer = self.adjust_learning_rate(optimizer, epoch)

        if not os.path.exists(os.path.join(self.save, self.model_name)):
            os.makedirs(os.path.join(self.save, self.model_name))
        torch.save(self.model.state_dict(), str(os.path.join(self.save, self.model_name) + '/HashNet.pth'))

    def test_hashnet(self, database_loader, test_loader, database_labels, test_labels,
                     num_database, num_test):
        self.model.eval()
        qB = self.generate_whole_code(test_loader, num_test)
        dB = self.generate_whole_code(database_loader, num_database)
        map_ = CalcMap(qB, dB, test_labels.numpy(), database_labels.numpy())
        print('Test_MAP(retrieval database): %3.5f' % (map_))

    def forward(self, x):
        x = self.model(x)
        return x
