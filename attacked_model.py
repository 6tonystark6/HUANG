import torch
import torch.nn as nn
import torch.nn.functional as F


class Attacked_Model(nn.Module):
    def __init__(self, method, dataset, bit, attacked_models_path, dataset_path):
        super(Attacked_Model, self).__init__()
        self.method = method
        self.dataset = dataset
        self.bit = bit
        vgg_path = dataset_path + 'imagenet-vgg-f.mat'
        if self.dataset == 'FLICKR':
            tag_dim = 1386
            num_label = 24
        if self.dataset == 'COCO':
            tag_dim = 1024
            num_label = 80
        if self.dataset == 'NUS':
            tag_dim = 1000
            num_label = 21

        if self.method == 'DPSH':
            DPSH_path = attacked_models_path + str(self.method) + '_' + self.dataset + '_' + str(self.bit) + '/DPSH.pth'
            from attacked_methods.DPSH.DPSH import DPSH
            batch_size, lr, backbone, n_epochs, wd, yita, save = 32, 0.05, 'VGG19', 100, 1e-5,50, 'attacked_models/'
            self.model_DPSH = DPSH(self.bit, batch_size, lr, backbone, self.dataset, n_epochs, wd, yita, save)
            self.model_DPSH.load(DPSH_path)
            self.model_DPSH.cuda().eval()
        if self.method == 'HashNet':
            HashNet_path = attacked_models_path + str(self.method) + '_' + self.dataset + '_' + str(self.bit) + '/HashNet.pth'
            from attacked_methods.HashNet.HashNet import HashNet
            batch_size, lr, backbone, n_epochs, wd, yita, save = 32, 0.05, 'VGG19', 100, 1e-5, 50, 'attacked_models/'
            self.model_hashnet = HashNet(self.bit, batch_size, lr, backbone, self.dataset, n_epochs, wd, yita, save)
            self.model_hashnet.load(HashNet_path)
            self.model_hashnet.cuda().eval()
        if self.method == 'CSQ':
            CSQ_path = attacked_models_path + str(self.method) + '_' + self.dataset + '_' + str(self.bit) + '/CSQ.pth'
            from attacked_methods.CSQ.CSQ import CSQ
            batch_size, lr, backbone, n_epochs, wd, yita, save = 32, 0.05, 'VGG19', 100, 1e-5, 50, 'attacked_models/'
            self.model_CSQ = CSQ(self.bit, batch_size, lr, backbone, self.dataset, n_epochs, wd, yita, save)
            self.model_CSQ.load(CSQ_path)
            self.model_CSQ.cuda().eval()

    def generate_image_feature(self, data_images):
        num_data = data_images.size(0)
        B = torch.zeros(num_data, self.bit)
        if self.method == 'DPSH':
            for i in range(num_data):
                output = self.model_DPSH.generate_code(data_images[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'HashNet':
            for i in range(num_data):
                output = self.model_hashnet.generate_code(data_images[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        if self.method == 'CSQ':
            for i in range(num_data):
                output = self.model_CSQ.generate_code(data_images[i].type(torch.float).unsqueeze(0).cuda())
                B[i, :] = output.cpu().data
        return B

    def generate_image_hashcode(self, data_images):
        num_data = data_images.size(0)
        outputs = []
        if self.method == 'DPSH':
            for i in range(num_data):
                output = self.model_DPSH.generate_code(data_images[i].unsqueeze(0).cuda())
                outputs.append(output)
        elif self.method == 'HashNet':
            for i in range(num_data):
                output = self.model_hashnet.generate_code(data_images[i].unsqueeze(0).cuda())
                outputs.append(output)
        elif self.method == 'CSQ':
            for i in range(num_data):
                output = self.model_CSQ.generate_code(data_images[i].unsqueeze(0).cuda())
                outputs.append(output)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Stack the outputs directly on GPU
        B = torch.stack(outputs, dim=0).squeeze(1)
        B.retain_grad()
        return F.tanh(B)

    def image_model(self, data_images):
        if self.method == 'DPSH':
            output = self.model_DPSH(data_images)
        if self.method == 'HashNet':
            output = self.model_hashnet(data_images)
        if self.method == 'CSQ':
            output = self.model_CSQ(data_images)
        return output