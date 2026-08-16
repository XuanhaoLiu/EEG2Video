import numpy as np 



'''
Description: 
Author: Zhou Tianyi
Date: 2025-03-24 11:43:25
LastEditTime: 2025-04-24 14:25:17
LastEditors:  
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import math
import os.path as osp

class Diffusion(object):
    def __init__(self, time_steps):
        self.time_steps = time_steps
        self.betas = self._linear_beta_schedule()

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # coeff1
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def _get_index_from_list(self, vals: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        
        batch_size = t.shape[0]  # batch_size
        out = vals.gather(-1, t.cpu())  # (batch_size, 1)

        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _linear_beta_schedule(self, start=0.0001, end=0.02) -> torch.Tensor:
        return torch.linspace(start, end, self.time_steps)

    def forward(self, x_0, dynamic_beta):
        b, f, c, h, w = x_0.shape
        t = torch.randint(0, self.time_steps, (b,)).cuda().long()

        diverse_noise = torch.randn_like(x_0).to(self.device)
        same_noise_i = torch.randn((b, 1, c, h, w)).to(self.device)
        same_noise = same_noise_i.repeat(1, f, 1, 1, 1)

        diverse_noise = diverse_noise * math.sqrt(1-dynamic_beta)
        same_noise = same_noise * math.sqrt(dynamic_beta)

        sqrt_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )

        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) \
               + sqrt_one_minus_alphas_cumprod_t.to(self.device) * (diverse_noise + same_noise).to(self.device)

import random
import os
def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')
seed = 3407
seed_everything(seed)
GT_label = np.array([[23, 22, 9, 6, 18,       14, 5, 36, 25, 19,      28, 35, 3, 16, 24,      40, 15, 27, 38, 33,
             34, 4, 39, 17, 1,       26, 20, 29, 13, 32,     37, 2, 11, 12, 30,      31, 8, 21, 7, 10, ],
            [27, 33, 22, 28, 31,     12, 38, 4, 18, 17,      35, 39, 40, 5, 24,      32, 15, 13, 2, 16,
 	         34, 25, 19, 30, 23,     3, 8, 29, 7, 20,        11, 14, 37, 6, 21,      1, 10, 36, 26, 9, ],
            [15, 36, 31, 1, 34,      3, 37, 12, 4, 5,        21, 24, 14, 16, 39,     20, 28, 29, 18, 32,
             2, 27, 8, 19, 13,       10, 30, 40, 17, 26,     11, 9, 33, 25, 35,      7, 38, 22, 23, 6,],
            [16, 28, 23, 1, 39,      10, 35, 14, 19, 27,     37, 31, 5, 18, 11,      25, 29, 13, 20, 24,
            7, 34, 26, 4, 40 ,       12, 8, 22, 21, 30,      17, 2, 38, 9,  3 ,      36, 33, 6, 32, 15,],
            [18, 29, 7, 35, 22  ,    19, 12, 36, 8, 15,      28, 1, 34, 23, 20 ,     13, 37, 9, 16, 30  ,
             2, 33, 27, 21, 14 ,     38, 10, 17, 31, 3,      24, 39, 11, 32, 4,      25, 40, 5, 26, 6 ,],
            [29, 16, 1, 22, 34,      39, 24, 10, 8, 35,      27, 31, 23, 17, 2,      15, 25, 40, 3, 36,
             26, 6, 14, 37, 9,       12, 19, 30, 5, 28,      32, 4, 13, 18, 21,      20, 7, 11, 33, 38],
            [38, 34, 40, 10, 28,     7, 1, 37, 22, 9,        16, 5, 12, 36, 20,      30, 6, 15, 35, 2,
             31, 26, 18, 24, 8,      3, 23, 19, 14, 13,      21, 4, 25, 11, 32,      17, 39, 29, 33, 27]
            ])
from einops import rearrange
chosed_label = [i for i in range(1,41)]
if __name__ == '__main__':
    latents = np.load('../Seq2Seq/latent_out_40_classes_test.npy')
    opt = np.load('classification_optical_flow_score.npy')#[200,]
    print(opt.shape)
    labels = np.where(opt >= 1.799, 1, 0)
    label = rearrange(labels,'(a b)  -> a b',a=40)
    indices = [list(GT_label[6]).index(element) for element in chosed_label]
    label2 = label[indices,:]
    
    label = rearrange(label2,'a b -> (a b)')
    print(label.shape)
    
    x_0 = latents
    x_0 = torch.from_numpy(x_0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_data =None
    for i in range(200):
        dynamic_beta=0.3 if labels[i] ==0 else 0.2
        diffusion = Diffusion(time_steps=500)
        out = diffusion.forward(x_0[i:i+1], dynamic_beta)
        
        if save_data is None:
            save_data = out
        else:
            save_data = torch.cat((save_data, out), dim=0)
    
    torch.save(save_data, f'40_classes_latent_add_noise.pt')
