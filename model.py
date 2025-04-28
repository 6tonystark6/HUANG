import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import math
import torch.nn.functional as F

from utils import spectral_norm as SpectralNorm
from utils import h_swish, h_sigmoid


# LabelNet
class LabelNet(nn.Module):
    def __init__(self, bit, num_classes):
        super(LabelNet, self).__init__()
        # 初始化函数，定义网络结构
        # bit: 位数
        # num_classes: 类别数
        self.curr_dim = 16  # 当前维度为16
        self.size = 32  # 图像大小为32x32
        # 定义特征提取层，将输入的类别特征映射为高维特征向量
        self.feature = nn.Sequential(
            nn.Linear(num_classes, 4096),  # 全连接层，将输入映射到4096维
            nn.ReLU(True),  # ReLU激活函数，增加网络的非线性特性
            nn.Linear(4096, self.curr_dim * self.size * self.size)  # 再次全连接，将特征映射到当前维度*图像大小*图像大小的维度
        )
        # 定义卷积层，用于处理输入的特征图
        conv2d = [
            nn.Conv2d(16, 32, 4, 2, 1),  # 卷积层：输入通道16，输出通道32，卷积核大小4x4，步长2，填充1
            nn.InstanceNorm2d(32),  # 实例归一化，增强模型的泛化能力
            nn.Tanh(),  # Tanh激活函数，用于增加网络的非线性特性
            nn.Conv2d(32, 64, 5, 1, 2),  # 卷积层：输入通道32，输出通道64，卷积核大小5x5，步长1，填充2
            nn.InstanceNorm2d(64),  # 实例归一化
            nn.Tanh()  # Tanh激活函数
        ]
        # 将卷积层组装成一个Sequential模块
        self.conv2d = nn.Sequential(*conv2d)

    def forward(self, label_feature):
        # 前向传播函数，定义了数据从输入到输出的流程
        # label_feature: 类别特征
        label_feature = self.feature(label_feature)  # 特征提取
        label_feature = label_feature.view(label_feature.size(0), self.curr_dim, self.size, self.size)  # 调整特征形状
        label_feature = self.conv2d(label_feature)  # 卷积处理
        return label_feature  # 返回处理后的特征图


class ImageNet(nn.Module):
    def __init__(self, dim_image, bit, num_classes):
        super(ImageNet, self).__init__()
        self.features = nn.Sequential(
            # 第一层卷积: 输入3通道图像, 输出32通道, 核大小7x7, 步长2, 填充3
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 输出尺寸: [batch_size, 32, 56, 56]
            # 第二层卷积: 输入32通道, 输出64通道, 核大小5x5, 步长1, 填充2
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 输出尺寸: [batch_size, 64, 28, 28]
            # 为了匹配TextNet的输出尺寸 [batch_size, 64, 16, 16], 添加更多的层
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 输出尺寸: [batch_size, 64, 14, 14]
            # 调整到16x16
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(16, 16), mode='bilinear', align_corners=True)  # 使用上采样至16x16
        )

    def forward(self, x):
        x = self.features(x)
        return x


# Progressive Fusion Module
class SemanticNet(nn.Module):
    def __init__(self, dim_image, bit, num_classes, channels=64, r=4):
        super(SemanticNet, self).__init__()
        self.labelnet = LabelNet(bit, num_classes)
        self.imagenet = ImageNet(dim_image, bit, num_classes)
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.Tanh()
        )
        self.hashing = nn.Sequential(nn.Linear(4096, bit), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(4096, num_classes), nn.Sigmoid())

    def forward(self, label_feature, image_feature):
        print(label_feature.size())
        label_feature = self.labelnet(label_feature)
        image_feature = self.imagenet(image_feature)
        xa = label_feature + image_feature
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = label_feature * wei + image_feature * (1 - wei)
        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        mixed_feature = label_feature * wei2 + image_feature * (1 - wei2)
        mixed_tensor = self.conv2d(mixed_feature)
        mixed_tensor = mixed_tensor.view(mixed_tensor.size(0), -1)
        mixed_hashcode = self.hashing(mixed_tensor)
        mixed_label = self.classifier(mixed_tensor)
        return mixed_feature, mixed_hashcode, mixed_label





# GAN Objectives
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=0.0, target_fake_label=1.0):
        # 初始化函数，定义了一个GAN损失函数
        # gan_mode: GAN的模式，包括'lsgan'、'vanilla'和'wgangp'
        # target_real_label: 真实标签值，默认为0.0
        # target_fake_label: 生成标签值，默认为1.0
        super(GANLoss, self).__init__()

        # 将真实标签值和生成标签值注册为buffer，方便后续使用
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        # 保存GAN的模式
        self.gan_mode = gan_mode

        # 根据GAN的模式选择相应的损失函数
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()  # 使用均方误差损失函数
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()  # 使用带Logits的二元交叉熵损失函数
        elif gan_mode in ['wgangp']:
            self.loss = None  # Wasserstein GAN损失函数，不需要损失函数
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)  # 抛出错误，指出未实现该GAN模式

    def get_target_tensor(self, label, target_is_real):
        # 根据是否是真实样本，获取目标张量
        if target_is_real:
            real_label = self.real_label.expand(label.size(0), 1)  # 将真实标签值扩展为与label相同的形状
            target_tensor = torch.cat([label, real_label], dim=-1)  # 将标签与真实标签值连接在一起，形成目标张量
        else:
            fake_label = self.fake_label.expand(label.size(0), 1)  # 将生成标签值扩展为与label相同的形状
            target_tensor = torch.cat([label, fake_label], dim=-1)  # 将标签与生成标签值连接在一起，形成目标张量
        return target_tensor

    def __call__(self, prediction, label, target_is_real):
        # 调用函数，计算损失值
        if self.gan_mode in ['lsgan', 'vanilla']:  # 如果是LSGAN或Vanilla GAN模式
            target_tensor = self.get_target_tensor(label, target_is_real)  # 获取目标张量
            loss = self.loss(prediction, target_tensor)  # 计算损失值
        elif self.gan_mode == 'wgangp':  # 如果是Wasserstein GAN模式
            if target_is_real:
                loss = -prediction.mean()  # 计算真实样本的损失值
            else:
                loss = prediction.mean()  # 计算生成样本的损失值
        return loss  # 返回计算得到的损失值


# Learning Rate Scheduler
def get_scheduler(optimizer, opt):
    # 定义一个函数用于获取学习率调整器（scheduler）
    # optimizer: 优化器
    # opt: 训练参数选项

    if opt.lr_policy == 'linear':  # 如果学习率策略为线性调整
        def lambda_rule(epoch):
            # 定义一个线性规则函数
            lr_l = 1.0 - max(0, epoch + opt.epoch_count -
                             opt.n_epochs) / float(opt.n_epochs_decay + 1)  # 计算学习率的衰减率
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)  # 使用LambdaLR进行学习率调整

    elif opt.lr_policy == 'step':  # 如果学习率策略为步进调整
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)  # 使用StepLR进行学习率调整，根据opt.lr_decay_iters和gamma确定调整方式

    elif opt.lr_policy == 'plateau':  # 如果学习率策略为plateau（基于性能）
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)  # 使用ReduceLROnPlateau进行学习率调整，根据模型性能调整学习率

    elif opt.lr_policy == 'cosine':  # 如果学习率策略为余弦退火调整
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.n_epochs,
                                                   eta_min=0)  # 使用CosineAnnealingLR进行学习率调整，根据余弦函数进行学习率退火

    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)  # 抛出错误，指出未实现该学习率策略

    return scheduler  # 返回学习率调整器
