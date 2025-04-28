import h5py
import torch


def load_dataset(path):

    Data = h5py.File(path)

    images = Data['IAll'][:]
    labels = Data['LAll'][:]
    tags = Data['TAll'][:]

    images = images.transpose(3, 2, 0, 1)
    labels = labels.transpose(1, 0)
    tags = tags.transpose(1, 0)

    Data.close()

    return images, tags, labels


def split_dataset(images, tags, labels, query_size, training_size, database_size):

    X = {}
    X['query'] = images[0: query_size]
    X['train'] = images[query_size: training_size + query_size]
    X['retrieval'] = images[query_size: query_size + database_size]

    Y = {}
    Y['query'] = tags[0: query_size]
    Y['train'] = tags[query_size: training_size + query_size]
    Y['retrieval'] = tags[query_size: query_size + database_size]

    L = {}
    L['query'] = labels[0: query_size]
    L['train'] = labels[query_size: training_size + query_size]
    L['retrieval'] = labels[query_size: query_size + database_size]

    return X, Y, L


def allocate_dataset(X, Y, L):

    train_images = torch.from_numpy(X['train'])
    train_texts = torch.from_numpy(Y['train'])
    train_labels = torch.from_numpy(L['train'])

    database_images = torch.from_numpy(X['retrieval'])
    database_texts = torch.from_numpy(Y['retrieval'])
    database_labels = torch.from_numpy(L['retrieval'])

    test_images = torch.from_numpy(X['query'])
    test_texts = torch.from_numpy(Y['query'])
    test_labels = torch.from_numpy(L['query'])

    return train_images, train_texts, train_labels, database_images, database_texts, database_labels, test_images, test_texts, test_labels


class Dataset_Config(object):
    def __init__(self, dataset, dataset_path):
        # 初始化数据集配置类
        # dataset: 数据集名称，包括'FLICKR'、'COCO'和'NUS'
        # dataset_path: 数据集路径

        self.dataset = dataset  # 设置数据集名称
        self.dataset_path = dataset_path  # 设置数据集路径
        self.vgg_path = self.dataset_path + 'imagenet-vgg-f.mat'  # 设置VGG模型路径

        if self.dataset == 'FLICKR':  # 如果数据集为FLICKR
            self.data_path = self.dataset_path + 'FLICKR-25K.mat'  # 设置FLICKR数据路径
            self.tag_dim = 1386  # 设置标签维度
            self.query_size = 2000  # 设置查询集大小
            self.training_size = 5000  # 设置训练集大小
            self.database_size = 18015  # 设置检索库大小
            self.num_label = 24  # 设置标签数量

        if self.dataset == 'COCO':  # 如果数据集为COCO
            self.data_path = self.dataset_path + 'MS-COCO.mat'  # 设置COCO数据路径
            self.tag_dim = 1024  # 设置标签维度
            self.query_size = 2000  # 设置查询集大小
            self.training_size = 10000  # 设置训练集大小
            self.database_size = 121287  # 设置检索库大小
            self.num_label = 80  # 设置标签数量

        if self.dataset == 'NUS':  # 如果数据集为NUS
            self.data_path = self.dataset_path + 'NUS-WIDE.mat'  # 设置NUS数据路径
            self.tag_dim = 1000  # 设置标签维度
            self.query_size = 2100  # 设置查询集大小
            self.training_size = 10500  # 设置训练集大小
            self.database_size = 193734  # 设置检索库大小
            self.num_label = 21  # 设置标签数量
