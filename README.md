# [HUANG: A Robust Diffusion Model-based Targeted Adversarial Attack Against Deep Hashing Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/32377) (AAAI 2025)
By Chihan Huang, Xiaobo Shen

Deep hashing models have achieved great success in retrieval tasks due to their powerful representation and strong information compression capabilities. However, they inherit the vulnerability of deep neural networks to adversarial perturbations. Attackers can severely impact the retrieval capability of hashing models by adding subtle, carefully crafted adversarial perturbations to benign images, transforming them into adversarial images. Most existing adversarial attacks target image classification models, with few focusing on retrieval models. We propose HUANG, the first targeted adversarial attack algorithm to leverage a diffusion model for hashing retrieval in black-box scenarios. In our approach, adversarial denoising uses adversarial perturbations and residual image to guide the shift from benign to adversarial distribution. Extensive experiments demonstrate the superiority of HUANG across different datasets, achieving state-of-the-art performance in black-box targeted attacks. Additionally, the dynamic interplay between denoising and adding adversarial perturbations in adversarial denoising endows HUANG with exceptional robustness and transferability.

## Code Organization
```
project-root/
├── Datasets/             # place to put dataset files
├── attacked_methods/     # attacked model folder
│   ├── CSQ               # CSQ
│   │   └── CSQ.py        # CSQ
│   ├── DPSH              # DPSH
│   │   └── DPSH.py       # DPSH
│   └── HashNet           # HashNet
│       └── HashNet.py    # HashNet
├── attacked_models/      # place to put trained hashing model files
├── checkpoint/           # place to put trained diffusion model checkpoints
├── data/                 # place to put datasets
├── models/               # diffusion model folder
│   ├── diffusion.py      # diffusion model
│   └── ema.py            # code for exponential moving average
├── runners/              # diffusion model folder
├── runs/                 # test image output
│   └── image_attack      # attack diffusion model
├── attack_model.py       # semantic net main code
├── attacked_model.py     # load attacked model
├── hashing_train.py      # train the attacked hashing model
├── load_data.py          # dataset configuration
├── main.py               # code for the attack
├── model.py              # components of semantic net
├── README.md             # readme
├── semanticnet_train.md  # code for training the semantic net
└── utils.py              # utils
```

# Requirements

- python == 3.8.20
- pytorch == 2.0.0+cu118
- torchvision == 0.15.0+cu118
- numpy == 1.24.4
- matplotlib == 3.7.5
- scipy == 1.10.1
- tqdm == 4.67.1
- pandas == 2.0.3
- PyYAML == 5.4.1
- tensorboard == 2.4.1
- triton == 2.0.0



## Use

### Train deep hashing models
Initialize the hyper-parameters in hashing_train.py, and then run
```Python
python hashing_train.py
```
and the checkpoints will be saved in '/attacked_models/DPSH_FLICKR_32/DPSH.pth', where DPSH and FLICKR are the attacked model and dataset you choose.

### Train semanticnet

```Python
python semanticnet_train.py --dataset FLICKR --attacked_method DPSH --bit 32 --batch_size 2 --learning_rate 1e-4 --output_dir DPSH_FLICKR_32
```

### Train diffusion model
Here is the [PyTorch implementation](https://github.com/ermongroup/ddim) for training the model.

### Attack

```Python
python main.py --dataset FLICKR --attacked_method DPSH --bit 32 --exp ./runs/ --sample -i images --npy_name attack --sample_step 3 --t 500  --ni
```


## Citation
```
@article{32377, 
    title={HUANG: A Robust Diffusion Model-based Targeted Adversarial Attack Against Deep Hashing Retrieval}, 
    volume={39}, 
    url={https://ojs.aaai.org/index.php/AAAI/article/view/32377}, 
    DOI={10.1609/aaai.v39i4.32377}, 
    number={4}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Huang, Chihan and Shen, Xiaobo}, 
    year={2025}, 
    month={Apr.}, 
    pages={3626-3634} 
}
```
