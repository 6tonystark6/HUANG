import argparse
import random
import traceback
import shutil
import logging

import yaml
import sys

from load_data import load_dataset, split_dataset, allocate_dataset, Dataset_Config

from runners.image_attack import Diffusion
from attacked_methods.DPSH.DPSH import *


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--dataset', dest='dataset', default='FLICKR', choices=['FLICKR', 'COCO', 'NUS'])
    parser.add_argument('--dataset_path', dest='dataset_path', default='./Datasets/', help='path of the dataset')
    parser.add_argument('--attacked_method', dest='attacked_method', default='DPSH', choices=['DPSH', 'HashNet', 'CSQ'])
    parser.add_argument('--attacked_models_path', dest='attacked_models_path', default='attacked_models/')
    parser.add_argument('--bit', dest='bit', type=int, default=32, choices=[16, 32, 48, 64])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--npy_name', type=str, required=True)
    parser.add_argument('--sample_step', type=int, default=3, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    DataConfigs = Dataset_Config(args.dataset, args.dataset_path)
    X, Y, L = load_dataset(DataConfigs.data_path)
    X_s, Y_s, L_s = split_dataset(X, Y, L, DataConfigs.query_size, DataConfigs.training_size, DataConfigs.database_size)
    image_train, _, label_train, image_database, _, label_database, image_test, _, label_test = allocate_dataset(X_s,
                                                                                                                 Y_s,
                                                                                                                 L_s)

    indices = random.sample(range(image_test.size(0)), config.sampling.batch_size)
    demo_tar_images = image_test[indices]
    demo_tar_labels = label_test[indices]
    demo_tar_images = demo_tar_images.float() / 255.0

    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print("<" * 80)

    try:
        runner = Diffusion(args, config)
        runner.image_attack_sample(tar_image=demo_tar_images, tar_label=demo_tar_labels)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
