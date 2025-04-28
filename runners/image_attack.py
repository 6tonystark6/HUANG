import os
import numpy as np
from torch import nn
from tqdm import tqdm
import scipy
import torch.nn.functional as F

import torch
import torchvision.utils as tvu
import warnings

warnings.filterwarnings('ignore')

from models.diffusion import Model
from attacked_model import Attacked_Model
from utils import calc_hamming_dist, calc_similarity_dist
from model import SemanticNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    # out = torch.gather(torch.tensor(a).to(t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *,
                                               model,
                                               logvar,
                                               betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


def image_editing_denoising_step_with_guidance(x, t, *,
                                               x_0,
                                               tar_img,
                                               tar_label,
                                               attacked_model,
                                               proto_net,
                                               model,
                                               logvar,
                                               betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    def hamming_grad_fn(x: torch.Tensor, tar_img: torch.Tensor):
        assert tar_img is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)  # torch.Size([8, 3, 224, 224])
            tar_img_in = tar_img.detach().requires_grad_(True)
            x_code = attacked_model.generate_image_hashcode(x_in)  # torch.Size([8, 32])
            tar_code = attacked_model.generate_image_hashcode(tar_img_in)  # torch.Size([8, 32])
            distH = calc_hamming_dist(x_code, tar_code)  # torch.Size([8, 8])
            distH.backward(torch.ones_like(distH))
            grad = x_in.grad
            return grad if grad is not None else 0

    def similarity_grad_fn(x, tar_img, tar_label):
        assert tar_img is not None
        assert tar_label is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)  # torch.Size([8, 3, 224, 224])
            tar_img_in = tar_img.detach().requires_grad_(True)  # torch.Size([8, 3, 224, 224])
            tar_label = tar_label.to(x_in.dtype)
            x_feat, _, _ = proto_net(tar_label, x_in)
            tar_feat, _, _ = proto_net(tar_label, tar_img_in)
            distS = F.cosine_similarity(x_feat, tar_feat, dim=1)
            grad = torch.autograd.grad(distS.sum(), x_in)[0].float()  # gradient ascend
            return grad

    grad_1 = hamming_grad_fn(x, tar_img)
    grad_2 = similarity_grad_fn(x, tar_img, tar_label)
    grad_2_upsampled = F.interpolate(grad_2, size=x.shape[2:], mode='bilinear', align_corners=False)
    residual_image = x - x_0

    logvar = extract(logvar, t, x.shape)
    mean = mean + 0.5 * (grad_1 + grad_2_upsampled) - torch.exp(0.5 * logvar) * residual_image

    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.bit = args.bit
        self.dim_image = config.data.image_size
        self.num_classes = config.data.num_label
        self.model_name = '{}_{}_{}'.format(args.attacked_method, args.dataset, args.bit)
        self._build_model(args)

    def _build_model(self, args):
        self.attacked_model = Attacked_Model(args.attacked_method, args.dataset, args.bit, args.attacked_models_path,
                                             args.dataset_path)
        self.attacked_model.eval()
        self.semanticnet = nn.DataParallel(SemanticNet(self.dim_image, self.bit, self.num_classes)).cuda()

    def load_semanticnet(self):
        self.semanticnet.module.load_state_dict(
            torch.load(os.path.join('outputs', self.model_name, 'Model', 'semanticnet_{}.pth'.format(self.model_name))))
        self.semanticnet.eval()

    def image_attack_sample(self, tar_image, tar_label):
        print("Loading model")

        model = Model(self.config)
        ckpt = torch.load('checkpoint/model.ckpt', map_location=self.device)
        model.load_state_dict(ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        self.load_semanticnet()
        print("Model loaded")
        ckpt_id = 0

        n = self.config.sampling.batch_size
        model.eval()
        print("Start sampling")
        with torch.no_grad():
            name = self.args.npy_name
            img = torch.load("data/{}.pth".format(name))[1]

            img = img.to(self.config.device)
            x0 = img

            tvu.save_image(x0, os.path.join(self.args.image_folder, f'original_input.png'))
            x0 = (x0 - 0.5) * 2.

            for it in range(self.args.sample_step):
                e = torch.randn_like(x0)  # noise
                total_noise_levels = self.args.t  # steps
                a = (1 - self.betas).cumprod(dim=0)  # \alpha_product
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()  # forward x_t
                tvu.save_image((x + 1) * 0.5,
                               os.path.join(self.args.image_folder, f'init_{ckpt_id}.png'))  # save noise image (x_T)

                with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
                    for i in reversed(range(total_noise_levels)):  # reverse sampling
                        t = (torch.ones(n) * i).to(self.device)  # current step
                        # x = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                        #                                                logvar=self.logvar,
                        #                                                betas=self.betas)
                        x = image_editing_denoising_step_with_guidance(x, t=t, x_0=x0, model=model,
                                                                       tar_img=tar_image, tar_label=tar_label,
                                                                       attacked_model=self.attacked_model,
                                                                       proto_net=self.semanticnet,
                                                                       logvar=self.logvar,
                                                                       betas=self.betas)
                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'noise_t_{i}_{it}.png'))
                        progress_bar.update(1)

                torch.save(x, os.path.join(self.args.image_folder,
                                           f'samples_{it}.pth'))
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'samples_{it}.png'))

