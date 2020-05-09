"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import torch.autograd as autograd
from torchvision.datasets import MNIST
from datasets.celeba import CelebA
from torchvision.datasets import CIFAR10
from utils import *
import numpy as np
from tensorboardX import SummaryWriter
import json 

import logging
import os
import time
import math
import argparse
import pprint


parser = argparse.ArgumentParser()
# action
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--visualize', action='store_true', help='Visualize manipulated attribures.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--seed', type=int, help='Random seed to use.')
# paths and reporting
parser.add_argument('--data_dir', default='/media/tiwari/My Passport/lokender/anish/normalizing_flows/datasets/data/', help='Location of datasets.')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--log_interval', type=int, default=2, help='How often to show loss statistics and save samples.')
parser.add_argument('--save_interval', type=int, default=50, help='How often to save during training.')
parser.add_argument('--eval_interval', type=int, default=1, help='Number of epochs to eval model and save model checkpoint.')
# data
parser.add_argument('--dataset', type=str, help='Which dataset to use.')
# model parameters
parser.add_argument('--depth', type=int, default=24, help='Depth of the network (cf Glow figure 2).')
parser.add_argument('--n_levels', type=int, default=3, help='Number of levels of of the network (cf Glow figure 2).')
parser.add_argument('--width', type=int, default=256, help='Dimension of the hidden layers.')
parser.add_argument('--z_std', type=float, help='Pass specific standard devition during generation/sampling.')
# training params
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
parser.add_argument('--batch_size_init', type=int, default=256, help='Batch size for the data dependent initialization.')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--n_epochs_warmup', type=int, default=2, help='Number of warmup epochs for linear learning rate annealing.')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--mini_data_size', type=int, default=None, help='Train only on this number of datapoints.')
parser.add_argument('--grad_norm_clip', default=50, type=float, help='Clip gradients during training.')
parser.add_argument('--checkpoint_grads', action='store_true', default=False, help='Whether to use gradient checkpointing in forward pass.')
parser.add_argument('--n_bits', default=5, type=int, help='Number of bits for input images.')
# distributed training params
parser.add_argument('--distributed', action='store_true', default=False, help='Whether to use DistributedDataParallels on multiple machines and GPUs.')
parser.add_argument('--world_size', type=int, default=1, help='Number of nodes for distributed training.')
parser.add_argument('--local_rank', type=int, default=1,help='When provided, run model on this cuda device. When None, used by torch.distributed.launch utility to manage multi-GPU training.')
# visualize
parser.add_argument('--vis_img', type=str, help='Path to image file to manipulate attributes and visualize.')
parser.add_argument('--vis_attrs', nargs='+', type=int, help='Which attribute to manipulate.')
parser.add_argument('--vis_alphas', nargs='+', type=float, help='Step size on the manipulation direction.')

parser.add_argument('--debug', action='store_true', default=False, help='Generates descriptive and verbose log file')
parser.add_argument('--debug_ac', action='store_true', default=False, help='Generates descriptive and verbose log file')

args = parser.parse_args()



best_eval_logprob = float('-inf')

# torch.autograd.set_detect_anomaly(True)
# --------------------
# Data
# --------------------

def fetch_dataloader(args, train=True, data_dependent_init=False):
    args.input_dims = {'mnist': (3,32,32), 'celeba': (3,64,64),'cifar':(3,32,32)}[args.dataset]

    transforms = {'mnist': T.Compose([T.Pad(2),                                         # image to 32x32 same as CIFAR
                                      T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # random shifts to fill the padded pixels
                                      T.ToTensor(),
                                      T.Lambda(lambda t: t + torch.rand_like(t)/2**8),  # dequantize
                                      T.Lambda(lambda t: t.expand(3,-1,-1))]),          # expand to 3 channels

                  'celeba': T.Compose([T.CenterCrop(148),  # RealNVP preprocessing
                                       T.Resize(64),
                                       T.Lambda(lambda im: np.array(im, dtype=np.float32)),                     # to numpy
                                       T.Lambda(lambda x: np.floor(x / 2**(8 - args.n_bits)) / 2**args.n_bits), # lower bits
                                       T.ToTensor(),  # note: if input to this transform is uint8, it divides by 255 and returns float
                                       T.Lambda(lambda t: t + torch.rand_like(t) / 2**args.n_bits)]),            # dequantize

                  'cifar': T.Compose([T.Lambda(lambda im: np.array(im, dtype=np.float32)),                     # to numpy
                                       T.Lambda(lambda x: np.floor(x / 2**(8 - args.n_bits)) / 2**args.n_bits), # lower bits
                                       T.ToTensor(),  # note: if input to this transform is uint8, it divides by 255 and returns float
                                       T.Lambda(lambda t: t + torch.rand_like(t) / 2**args.n_bits)])
                  }[args.dataset]

    dataset = {'mnist': MNIST, 'celeba': CelebA,'cifar':CIFAR10}[args.dataset]

    # load the specific dataset
    if args.dataset=='cifar':
        dataset = dataset(root=args.data_dir, train=train, transform=transforms,download=True)
    else:
        dataset = dataset(root=args.data_dir, train=train, transform=transforms)



    if args.mini_data_size:
        dataset.data = dataset.data[:args.mini_data_size]

    # load sampler and dataloader
    if args.distributed and train is True and not data_dependent_init:  # distributed training; but exclude initialization
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    batch_size = args.batch_size_init if data_dependent_init else args.batch_size  # if data dependent init use init batch size
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device.type is 'cuda' else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), drop_last=True, sampler=sampler, **kwargs)


# --------------------
# Model component layers
# --------------------

class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,3,1,1),topmost=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())
        self.topmost=topmost
    def forward(self, x):
        # print('actnorm c',c)
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, False) + 1e-6).view_as(self.bias)
            self.initialized += 1

        z = (x - self.bias) / self.scale
        logdet = - self.scale.abs().log().sum() * x.shape[2] * x.shape[3]
        return z, logdet

    def inverse(self, z):
        return z * self.scale + self.bias, self.scale.abs().log().sum() * z.shape[2] * z.shape[3]


class Invertible1x1Conv(nn.Module):
    """ Invertible 1x1 convolution layer; cf Glow section 3.2 """
    def __init__(self, n_channels=3, lu_factorize=False,topmost=False):
        super().__init__()
        self.lu_factorize = lu_factorize
        self.topmost=topmost

        # initiaize a 1x1 convolution weight matrix
        w = torch.randn(n_channels, n_channels).to(args.device)
        w = torch.qr(w)[0]  # note: nn.init.orthogonal_ returns orth matrices with dets +/- 1 which complicates the inverse call below

        if lu_factorize:
            # compute LU factorization
            # p, l, u = torch.btriunpack(*w.unsqueeze(0).btrifact())
            p, l, u = torch.lu_unpack(*w.unsqueeze(0).lu())

            # initialize model parameters
            self.p, self.l, self.u = nn.Parameter(p.squeeze()), nn.Parameter(l.squeeze()), nn.Parameter(u.squeeze())
            s = self.u.diag()
            self.log_s = nn.Parameter(s.abs().log())
            self.register_buffer('sign_s', s.sign())  # note: not optimizing the sign; det W remains the same sign
            self.register_buffer('l_mask', torch.tril(torch.ones_like(self.l), -1))  # store mask to compute LU in forward/inverse pass
        else:
            self.w = nn.Parameter(w)

    def forward(self, x):
        B,C,H,W = x.shape
        if self.lu_factorize:
            l = self.l * self.l_mask + torch.eye(C).to(self.l.device)
            u = self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp())
            self.w = self.p @ l @ u
            logdet = self.log_s.sum() * H * W
        else:
            logdet = torch.slogdet(self.w)[-1] * H * W

        return F.conv2d(x, self.w.view(C,C,1,1)), logdet

    def inverse(self, z):
        B,C,H,W = z.shape
        if self.lu_factorize:
            l = torch.inverse(self.l * self.l_mask + torch.eye(C).to(self.l.device))
            u = torch.inverse(self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp()))
            w_inv = u @ l @ self.p.inverse()
            logdet = - self.log_s.sum() * H * W
        else:
            w_inv = self.w.inverse()
            logdet = - torch.slogdet(self.w)[-1] * H * W

        return F.conv2d(z, w_inv.view(C,C,1,1)), logdet


class AffineCoupling(nn.Module):
    """ Affine coupling layer; cf Glow section 3.3; RealNVP figure 2 """
    def __init__(self, n_channels, width,topmost=False,num_labels=10):
        super().__init__()
        # network layers;
        # per realnvp, network splits input, operates on half of it, and returns shift and scale of dim = half the input channels
        self.conv1 = nn.Conv2d(n_channels//2, width, kernel_size=3, padding=1, bias=False)  # input is split along channel dim
        self.actnorm1 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv2 = nn.Conv2d(width, width, kernel_size=1, padding=1, bias=False)
        self.actnorm2 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv3 = nn.Conv2d(width, n_channels, kernel_size=3)            # output is split into scale and shift components
        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels,1,1))   # learned scale (cf RealNVP sec 4.1 / Glow official code
        self.topmost=topmost
        # print(self.topmost)
        if topmost:
            self.c_dim = 12
            self.fc = nn.Linear(num_labels,48*2*2,bias=False)
            # self.fc2 = nn.Linear((self.c_dim+48)*2*2,48*2*2,bias=False)
            # self.actnorm3 = Actnorm(param_dim=(1,n_channels//2,1,1))
            # self.bn = nn.BatchNorm2d(48)
            # self.fc.weight.data.zero_()
            # self.fc2.weight.data.zero_()
        # if topmost:
        #     self.linear1 = nn.Linear()
        # initialize last convolution with zeros, such that each affine coupling layer performs an identity function
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()



    def forward(self, x,c=None):
        x_a, x_b = x.chunk(2, 1)  # split along channel dim
        

        # if self.topmost:
        # print(x.shape)
        # print('Affine layer c',c)
        # print(self.topmost,'\n')

        if self.topmost:
            # c=  c.long()
            lin_c = F.relu(self.fc(c))
            # lin_c = lin_c.view(-1,self.c_dim,2,2)
            lin_c = lin_c.view(-1,48,2,2)
            x_b = x_b+lin_c
            # # print(x_b.shape,lin_c.shape)
            # x_b = torch.cat([x_b,lin_c],dim=1)
            # # print(x_b.shape)
            # x_b = x_b.view(-1,(self.c_dim+48)*2*2)
            # x_b = self.fc2(x_b)
            # x_b = x_b.view(-1,48,2,2)
            # # x_b = self.actnorm3(x_b)[0]
            # x_b = self.bn(x_b)
            # x_b = F.relu(x_b)

            # print(x_b.shape)

            # print('XA,XB shape',x_a.shape,x_b.shape)
        # print(x_b.shape)
        h = F.relu(self.actnorm1(self.conv1(x_b))[0])
        # print('h',h.shape)
        h = F.relu(self.actnorm2(self.conv2(h))[0])
        # print('h2',h.shape)
        h = self.conv3(h) * self.log_scale_factor.exp()
        # h = self.conv3(h)

        # print('h3',h.shape)
        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        s = torch.sigmoid(s + 2.)  # at initalization, s is 0 and sigmoid(2) is near identity
        # s = torch.tanh(s + 2.)  # at initalization, s is 0 and sigmoid(2) is near identity
        # scale_factor = self.log_scale_factor.exp()
        # scale_factor = scale_factor+1e-4
        # # scale_factor += scale_factor.mean(0)*1e-3
        # if torch.isnan(scale_factor).any():
        #     raise RuntimeError('Scale factor has NaN entries')
        # s = torch.tanh(s+2.) * scale_factor
        # ss = np.linalg.norm(s.detach().cpu().numpy().flatten())
        # logging.info(ss)
        z_a = s * x_a + t

        z_b = x_b
        z = torch.cat([z_a, z_b], dim=1)  # concat along channel dim

        logdet = s.log().sum([1, 2, 3])
        if args.debug_ac:
            if self.topmost:
                logging.info('Inside AC  with class , sum_logdet:{}'.format(logdet.cpu().mean(0).item()))
            else:
                logging.info('Inside AC , sum_logdet:{}'.format(logdet.cpu().mean(0).item()))


        return z, logdet

    def inverse(self, z,c=None):
        z_a, z_b = z.chunk(2, 1)  # split along channel dim
        if self.topmost:
            # c=  c.long()
            lin_c = F.relu(self.fc(c))
            # lin_c = lin_c.view(-1,self.c_dim,2,2)
            lin_c = lin_c.view(-1,48,2,2)
            z_b = z_b+lin_c
            # lin_c = F.relu(self.fc(c))
            # lin_c = lin_c.view(-1,self.c_dim,2,2)
            # # print(x_b.shape,lin_c.shape)
            # z_b = torch.cat([z_b,lin_c],dim=1)
            # # print(x_b.shape)
            # z_b = z_b.view(-1,(self.c_dim+48)*2*2)
            # z_b = self.fc2(z_b)
            # z_b = z_b.view(-1,48,2,2)
            # # z_b = self.actnorm3(z_b)[0]
            # z_b = self.bn(z_b)
        
            # z_b = F.relu(z_b)

        # print('ZA,ZB shape',z_a.shape,z_b.shape)
        h = F.relu(self.actnorm1(self.conv1(z_b))[0])
        h = F.relu(self.actnorm2(self.conv2(h))[0])
        h = self.conv3(h)  * self.log_scale_factor.exp()
        # h = self.conv3(h)
        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels

        # scale_factor = self.log_scale_factor.exp()
        # # print(scale_factor.mean(0))
        # scale_factor = scale_factor+1e-4


        # if torch.isnan(scale_factor).any():
        #     raise RuntimeError('Scale factor has NaN entries')
        # s = torch.tanh(s+2.) * scale_factor
        s = torch.sigmoid(s + 2.)

        x_a = (z_a - t) / s
        x_b = z_b
        x = torch.cat([x_a, x_b], dim=1)  # concat along channel dim

        logdet = - s.log().sum([1, 2, 3])
        if args.debug_ac:
            if self.topmost:
                logging.info('Inside AC inverse with class , sum_logdet:{}'.format(logdet.cpu().mean(0).item()))
            else:
                logging.info('Inside AC inverse , sum_logdet:{}'.format(logdet.cpu().mean(0).item()))



        return x, logdet




class Squeeze(nn.Module):
    """ RealNVP squeezing operation layer (cf RealNVP section 3.6; Glow figure 2b):
    For each channel, it divides the image into subsquares of shape 2 × 2 × c, then reshapes them into subsquares of 
    shape 1 × 1 × 4c. The squeezing operation transforms an s × s × c tensor into an s/2 × s/2 × 4c tensor """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C, H//2, 2, W//2, 2)   # factor spatial dim
        x = x.permute(0, 1, 3, 5, 2, 4)         # transpose to (B, C, 2, 2, H//2, W//2)
        x = x.reshape(B, 4*C, H//2, W//2)       # aggregate spatial dim factors into channels
        return x

    def inverse(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C//4, 2, 2, H, W)      # factor channel dim
        x = x.permute(0, 1, 4, 2, 5, 3)         # transpose to (B, C//4, H, 2, W, 2)
        x = x.reshape(B, C//4, 2*H, 2*W)        # aggregate channel dim factors into spatial dims
        return x


class Split(nn.Module):
    """ Split layer; cf Glow figure 2 / RealNVP figure 4b
    Based on RealNVP multi-scale architecture: splits an input in half along the channel dim; half the vars are
    directly modeled as Gaussians while the other half undergo further transformations (cf RealNVP figure 4b).
    """
    def __init__(self, n_channels):
        super().__init__()
        self.gaussianize = Gaussianize(n_channels//2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim
        z2, logdet = self.gaussianize(x1, x2)
        return x1, z2, logdet

    def inverse(self, x1, z2):
        x2, logdet = self.gaussianize.inverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x, logdet


class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Conv2d(n_channels, 2*n_channels, kernel_size=3, padding=1)  # computes the parameters of Gaussian
        self.log_scale_factor = nn.Parameter(torch.zeros(2*n_channels,1,1))       # learned scale (cf RealNVP sec 4.1 / Glow official code
        # initialize to identity
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2):
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]          # split along channel dims
        z2 = (x2 - m) * torch.exp(-logs)                # center and scale; log prob is computed at the model forward
        logdet = - logs.sum([1,2,3])
        return z2, logdet

    def inverse(self, x1, z2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1,2,3])
        return x2, logdet


class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        logdet = - math.log(256) * x[0].numel() # processing each image dim from [0, 255] to [0,1]; per RealNVP sec 4.1 taken into account
        return x - 0.5, logdet                  # center x at 0

    def inverse(self, x):
        logdet = math.log(256) * x[0].numel()
        return x + 0.5, logdet

# --------------------
# Container layers
# --------------------

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def __init__(self, *args, **kwargs):
        self.topmost = kwargs.pop('topmost',True)
        # self.topmost=topmost

        self.checkpoint_grads = kwargs.pop('checkpoint_grads', None)
        super().__init__(*args, **kwargs)

    def forward(self, x,c=None):
        sum_logdets = 0.
        # print('C is not None; self.topmost',(c is not None),self.topmost)
        # assert((c is not None)==self.topmost)
        # print('self',self,'\n')
        for idx,module in enumerate(self):
            if (isinstance(module,FlowStep) or isinstance(module,AffineCoupling)) :# and self.topmost:
                # print(self.topmost,c)
                # print('in here',module)
                x, logdet = module(x,c=c) if not self.checkpoint_grads else checkpoint(module, x,c=c)
                # logging.info('Inside FlowStep with class , sum_logdet:{}'.format(logdet.cpu().mean(0).item()))

                # print('x logdet',x.shape,logdet.shape)

            else:
                # print(module)
                # print(self.topmost,c)
                # print(module)
                x, logdet = module(x) if not self.checkpoint_grads else checkpoint(module, x)
                # logging.info('Inside FlowStep  , sum_logdet:{}'.format(logdet.cpu().mean(0).item()))

                # print('x logdet',x.shape,logdet.shape)
            sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    def inverse(self, z,c=None):
        sum_logdets = 0.
        for module in reversed(self):
            if (isinstance(module,FlowStep) or isinstance(module,AffineCoupling)):# and self.topmost:
                # print(self.topmost,c)
                # print('in here',module)
                z, logdet = module.inverse(z,c=c)
                # if c is not None:
                #     logging.info('Inside FlowStep inverse with class , sum_logdet:{}'.format(logdet.cpu().mean(0).item()))

                # print('x logdet',x.shape,logdet.shape)

            else:
                # print(module)
                # print(self.topmost,c)

                z, logdet = module.inverse(z)
                # logging.info('Inside FlowStep  , sum_logdet:{}'.format(logdet.cpu().mean(0).item()))

            # if isinstance(module,AffineCoupling) and self.topmost:
            #     z, logdet = module.inverse(z,c)
            # else:
            #     z, logdet = module.inverse(z)
            sum_logdets = sum_logdets + logdet
        return z, sum_logdets


class FlowStep(FlowSequential):
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """
    def __init__(self, n_channels, width, lu_factorize=False,topmost=False):
        super().__init__(Actnorm(param_dim=(1,n_channels,1,1),topmost=topmost),
                         Invertible1x1Conv(n_channels, lu_factorize,topmost=topmost),
                         AffineCoupling(n_channels, width,topmost=topmost))


class FlowLevel(nn.Module):
    """ One depth level of Glow flow (Squeeze -> FlowStep x K -> Split); cf Glow figure 2b """
    def __init__(self, n_channels, width, depth, checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        # network layers
        self.squeeze = Squeeze()
        self.flowsteps = FlowSequential(*[FlowStep(4*n_channels, width, lu_factorize) for _ in range(depth)], checkpoint_grads=checkpoint_grads)
        self.split = Split(4*n_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x, logdet_flowsteps = self.flowsteps(x)
        if args.debug:
            logging.info('Inside FlowLevel forward:, sum_logdet:{}'.format(logdet_flowsteps.cpu().mean(0).item()))

        x1, z2, logdet_split = self.split(x)
        logdet = logdet_flowsteps + logdet_split
        return x1, z2, logdet

    def inverse(self, x1, z2):
        x, logdet_split = self.split.inverse(x1, z2)
        x, logdet_flowsteps = self.flowsteps.inverse(x)
        if args.debug:
            logging.info('Inside FlowLevel inverse:, sum_logdet:{}'.format(logdet_flowsteps.cpu().mean(0).item()))
        x = self.squeeze.inverse(x)
        logdet = logdet_flowsteps + logdet_split
        return x, logdet


# --------------------
# Model
# --------------------

class Glow(nn.Module):
    """ Glow multi-scale architecture with depth of flow K and number of levels L; cf Glow figure 2; section 3"""
    def __init__(self, width, depth, n_levels, input_dims=(3,32,32), checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        # calculate output dims
        in_channels, H, W = input_dims
        out_channels = int(in_channels * 4**(n_levels+1) / 2**n_levels)  # each Squeeze results in 4x in_channels (cf RealNVP section 3.6); each Split in 1/2x in_channels
        out_HW = int(H / 2**(n_levels+1))                                # each Squeeze is 1/2x HW dim (cf RealNVP section 3.6)
        self.output_dims = out_channels, out_HW, out_HW

        # preprocess images
        self.preprocess = Preprocess()

        # network layers cf Glow figure 2b: (Squeeze -> FlowStep x depth -> Split) x n_levels -> Squeeze -> FlowStep x depth
        self.flowlevels = nn.ModuleList([FlowLevel(in_channels * 2**i, width, depth, checkpoint_grads, lu_factorize) for i in range(n_levels)])
        self.squeeze = Squeeze()
        final_flowstep = []
        for idx,_ in enumerate(range(depth)):
            # if idx==depth-1:
            final_flowstep.append(FlowStep(out_channels, width, lu_factorize,topmost=True))
            # else:
            #     final_flowstep.append(FlowStep(out_channels, width, lu_factorize,topmost=False))

        self.flowstep = FlowSequential(*final_flowstep, checkpoint_grads=checkpoint_grads,topmost=True)

        # gaussianize the final z output; initialize to identity
        self.gaussianize = Gaussianize(out_channels)

        # base distribution of the flow
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))
        # self.register_buffer('base_dist_var', torch.Tensor([2]))


    def forward(self, x,c=None):
        x, sum_logdets = self.preprocess(x)
        # pass through flow
        zs = []
        for m in self.flowlevels:
            # print('\n')
            x, z, logdet = m(x)
            sum_logdets = sum_logdets + logdet
            zs.append(z)
        x = self.squeeze(x)
        # print(zs[0].shape)
        if args.debug:
            logging.info('Before Flowstep:, sum_logdet:{}'.format(sum_logdets.cpu().mean(0).item()))
        z, logdet = self.flowstep(x,c)
        
        sum_logdets = sum_logdets + logdet
        if args.debug:
            logging.info('After Flowstep, sum_logdet:{}'.format(sum_logdets.cpu().mean(0).item()))
        logging.info('\n')
        # gaussianize the final z
        z, logdet = self.gaussianize(torch.zeros_like(z), z)
        sum_logdets = sum_logdets + logdet
        zs.append(z)
        return zs, sum_logdets

    def inverse(self,c, zs=None, batch_size=None, z_std=1.):
        if zs is None:  # if no random numbers are passed, generate new from the base distribution
            assert batch_size is not None, 'Must either specify batch_size or pass a batch of z random numbers.'
            zs = [z_std * self.base_dist.sample((batch_size, *self.output_dims)).squeeze()]
        # pass through inverse flow
        z, sum_logdets = self.gaussianize.inverse(torch.zeros_like(zs[-1]), zs[-1])
        if args.debug:
            logging.info('Before inverse Flowstep:, sum_logdet:{}'.format(sum_logdets.cpu().mean(0).item()))
        x, logdet = self.flowstep.inverse(z,c)
        sum_logdets = sum_logdets + logdet

        if args.debug:
            logging.info('After inverse Flowstep:, sum_logdet:{}'.format(sum_logdets.cpu().mean(0).item()))

        x = self.squeeze.inverse(x)
        for i, m in enumerate(reversed(self.flowlevels)):
            z = z_std * (self.base_dist.sample(x.shape).squeeze() if len(zs)==1 else zs[-i-2])  # if no z's are passed, generate new random numbers from the base dist
            x, logdet = m.inverse(x, z)
            sum_logdets = sum_logdets + logdet
        # postprocess
        x, logdet = self.preprocess.inverse(x)
        sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def log_prob(self, x, c,bits_per_pixel=False):
        zs, logdet = self.forward(x,c)

        logging.info('Logdet:{}'.format(logdet.cpu().mean(0).item()))
        log_prob = sum(self.base_dist.log_prob(z).sum([1,2,3]) for z in zs) + logdet
        logging.info('log prob no bits div:{}'.format(log_prob.cpu().mean(0).item()))
        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel())
        return log_prob

# --------------------
# Train and evaluate
# --------------------

@torch.no_grad()
def data_dependent_init(model, args):
    # set up an iterator with batch size = batch_size_init and run through model
    dataloader = fetch_dataloader(args, train=True, data_dependent_init=True)
    # print(next(iter(dataloader))[0].shape,next(iter))
    data_img,data_label = next(iter(dataloader))
    # print(data_img.shape,data_label.shape)
    # data_label = data_label.float()
    # print(data_label.shape)
    # print(data_label.type())
    data_label = one_hot_encode(data_label).to(args.device)
    # print('data label shape',data_label.shape)
    model(data_img.requires_grad_(True if args.checkpoint_grads else False).to(args.device),data_label.requires_grad_(True if args.checkpoint_grads else False).to(args.device))
    del dataloader
    return True

def train_epoch(model, dataloader, optimizer, writer, epoch, args):
    model.train()

    tic = time.time()

    for i, (x,y) in enumerate(dataloader):
        # print('Dataloader step ',i,'........................................................................')
        args.step += args.world_size
        # warmup learning rate
        if epoch <= args.n_epochs_warmup:
            optimizer.param_groups[0]['lr'] = args.lr * min(1, args.step / (len(dataloader) * args.world_size * args.n_epochs_warmup))
        # with autograd.detect_anomaly():
        x = x.requires_grad_(True if args.checkpoint_grads else False).to(args.device)  # requires_grad needed for checkpointing
        if torch.isnan(x).any().item():
            logging.info('Element of X is nan')
            logging.info('X :{}'.format(x))
            logging.info('Loss :{}'.format(loss))

        # print('Y',y,y.shape)

        y = one_hot_encode(y).to(args.device)
        y.requires_grad_()
        # print('Y',y,y.shape)
    
        loss = - model.log_prob(x, y, bits_per_pixel=True).mean(0)
        logging.info('Loss :{}'.format(loss))

        # print('Loss',loss,i)
        optimizer.zero_grad()
        loss.backward()
        if i % (args.log_interval+10) == 0:
            total_norm = 0.0
            for p in model.parameters():

                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            logging.info('Norm before clipping:{}'.format(total_norm))

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

        if i % (args.log_interval+10) == 0:
            total_norm = 0.0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            logging.info('Norm after clipping:{}'.format(total_norm))
        optimizer.step()
        # print('Optimized....................................................................................................')
        # report stats
        if i % args.log_interval == 0:
            # compute KL divergence between base and each of the z's that the model produces
            with torch.no_grad():
                zs, _ = model(x,y)
                kls = [D.kl.kl_divergence(D.Normal(z.mean(), z.std()), model.base_dist) for z in zs]

            # write stats
            if args.on_main_process:
                et = time.time() - tic              # elapsed time
                tt = len(dataloader) * et / (i+1)   # total time per epoch
                logging.info('Epoch: [{}/{}][{}/{}]\tStep: {}\tTime: elapsed {:.0f}m{:02.0f}s / total {:.0f}m{:02.0f}s\tLoss {:.4f}\t'.format(
                      epoch, args.start_epoch + args.n_epochs, i+1, len(dataloader), args.step, et//60, et%60, tt//60, tt%60, loss.item()))
                # update writer
                for j, kl in enumerate(kls):
                    writer.add_scalar('kl_level_{}'.format(j), kl.item(), args.step)
                writer.add_scalar('train_bits_x', loss.item(), args.step)

        # save and generate
        if i % args.save_interval == 0:
            # print('Generating................................................................................................')
            # generate samplesmake_grid

            samples,sample_labels = generate(model, n_samples=6, z_stds=[0., 0.25, 0.5, 0.7, 1.0])
            label_names_map_cifar =  {0 : 'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

            sample_labels = list(sample_labels.cpu().numpy().nonzero()[1])
            if args.dataset=='cifar':
                save_str = '_'.join(label_names_map_cifar[x] for x in sample_labels)
            else:
                save_str = '_'.join(str(x) for x in sample_labels)
            logging.info('Sample labels:{}'.format(save_str))

            images = make_grid(samples.cpu(), nrow=6, pad_value=1)

            # write stats and save checkpoints
            if args.on_main_process:
                save_image(images, os.path.join(args.output_dir, 'generated_sample_{}_{}.png'.format(args.step,save_str)))

                # save training checkpoint
                torch.save({'epoch': epoch,
                            'global_step': args.step,
                            'state_dict': model.state_dict()},
                            os.path.join(args.output_dir, 'checkpoint_ep_'+str(epoch)+'.pt'))
                torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))



@torch.no_grad()
def evaluate(model, dataloader, args):
    model.eval()
    print('Evaluating ...', end='\r')

    logprobs = []
    for x,y in dataloader:
        x = x.to(args.device)
        logprobs.append(model.log_prob(x, bits_per_pixel=True))
    logprobs = torch.cat(logprobs, dim=0).to(args.device)
    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.std(0) / math.sqrt(len(dataloader.dataset))
    return logprob_mean, logprob_std

@torch.no_grad()
def generate(model, n_samples, z_stds):
    model.eval()
    # print('Generating ...', end='\r')
    logging.info('Generating ...')

    samples = []
    rand_labels = get_random_labels(n_samples).to(args.device)

    for z_std in z_stds:
        sample, _ = model.inverse(rand_labels,batch_size=n_samples, z_std=z_std)
        log_probs = model.log_prob(sample, rand_labels,bits_per_pixel=True)
        samples.append(sample[log_probs.argsort().flip(0)])  # sort by log_prob; flip high (left) to low (right)
    return torch.cat(samples,0),rand_labels

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, writer, args):
    global best_eval_logprob

    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        train_epoch(model, train_dataloader, optimizer, writer, epoch, args)

        # evaluate
        if False:#epoch % args.eval_interval == 0:
            eval_logprob_mean, eval_logprob_std = evaluate(model, test_dataloader, args)
            logging.info('Evaluate at epoch {}: bits_x = {:.3f} +/- {:.3f}'.format(epoch, eval_logprob_mean, eval_logprob_std))

            # save best state
            if args.on_main_process:
            	if eval_logprob_mean > best_eval_logprob:
	                best_eval_logprob = eval_logprob_mean
	                torch.save({'epoch': epoch,
	                            'global_step': args.step,
	                            'state_dict': model.state_dict()},
	                            os.path.join(args.output_dir, 'best_model_checkpoint.pt'))


# --------------------
# Visualizations
# --------------------

def encode_dataset(model, dataloader):
    model.eval()

    zs = []
    attrs = []
    for i, (x,y) in enumerate(dataloader):
        print('Encoding [{}/{}]'.format(i+1, len(dataloader)), end='\r')
        x = x.to(args.device)
        zs_i, _ = model(x)
        zs.append(torch.cat([z.flatten(1) for z in zs_i], dim=1))
        attrs.append(y)

    zs = torch.cat(zs, dim=0)
    attrs = torch.cat(attrs, dim=0)
    print('Encoding completed.')
    return zs, attrs

def compute_dz(zs, attrs, idx):
    """ for a given attribute idx, compute the mean for all encoded z's corresponding to the positive and negative attribute """
    z_pos = [zs[i] for i in range(len(zs)) if attrs[i][idx] == +1]
    z_neg = [zs[i] for i in range(len(zs)) if attrs[i][idx] == -1]
    # dz = z_pos - z_neg; where z_pos is mean of all encoded datapoints where attr is present;
    return torch.stack(z_pos).mean(0) - torch.stack(z_neg).mean(0)   # out tensor of shape (flattened zs dim,)

def get_manipulators(zs, attrs):
    """ compute dz (= z_pos - z_neg) for each attribute """
    print('Extracting manipulators...', end=' ')
    dzs = 1.6 * torch.stack([compute_dz(zs, attrs, i) for i in range(attrs.shape[1])], dim=0)  # compute dz for each attribute official code multiplies by 1.6 scalar here
    print('Completed.')
    return dzs  # out (n_attributes, flattened zs dim)

def manipulate(model, z, dz, z_std, alpha):
    # 1. record incoming shapes
    z_dims   = [z_.squeeze().shape   for z_ in z]
    z_numels = [z_.numel() for z_ in z]
    # 2. flatten z into a vector and manipulate by alpha in the direction of dz
    z = torch.cat([z_.flatten(1) for z_ in z], dim=1).to(dz.device)
    z = z + dz * torch.tensor(alpha).float().view(-1,1).to(dz.device)  # out (n_alphas, flattened zs dim)
    # 3. reshape back to z shapes from each level of the model
    zs = [z_.view((len(alpha), *dim)) for z_, dim in zip(z.split(z_numels, dim=1), z_dims)]
    # 4. decode
    return model.inverse(zs, z_std=z_std)[0]

def load_manipulators(model, args):
    # construct dataloader with limited number of images
    args.mini_data_size = 30000
    # load z manipulators for each attribute
    if os.path.exists(os.path.join(args.output_dir, 'z_manipulate.pt')):
        z_manipulate = torch.load(os.path.join(args.output_dir, 'z_manipulate.pt'), map_location=args.device)
    else:
        # encode dataset, compute manipulators, store zs, attributes, and dzs
        dataloader = fetch_dataloader(args, train=True)
        zs, attrs = encode_dataset(model, dataloader)
        z_manipulate = get_manipulators(zs, attrs)
        torch.save(zs, os.path.join(args.output_dir, 'zs.pt'))
        torch.save(attrs, os.path.join(args.output_dir, 'attrs.pt'))
        torch.save(z_manipulate, os.path.join(args.output_dir, 'z_manipulate.pt'))
    return z_manipulate

@torch.no_grad()
def visualize(model, args, attrs=None, alphas=None, img_path=None, n_examples=1):
    """ manipulate an input image along a given attribute """
    dataset = fetch_dataloader(args, train=False).dataset  # pull the dataset to access transforms and attrs
    # if no attrs passed, manipulate all of them
    if not attrs:
        attrs = list(range(len(dataset.attr_names)))
    # if image is passed, manipulate only the image
    if img_path:
        from PIL import Image
        img = Image.open(img_path)
        x = dataset.transform(img)  # transform image to tensor and encode
    else:  # take first n_examples from the dataset
        x, _ = dataset[0]
    z, _ = model(x.unsqueeze(0).to(args.device))
    # get manipulors
    z_manipulate = load_manipulators(model, args)
    # decode the varied attributes
    dec_x =[]
    for attr_idx in attrs:
        dec_x.append(manipulate(model, z, z_manipulate[attr_idx].unsqueeze(0), args.z_std, alphas))
    return torch.stack(dec_x).cpu()


# --------------------
# Main
# --------------------

if __name__ == '__main__':
    
    args.step = 0  # global step
    args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else os.path.join(args.output_dir, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    writer = None  # init as None in case of multiprocessing; only main process performs write ops
    os.makedirs(args.output_dir,exist_ok=True)
    log_format = '%(levelname)-8s %(message)s'

    log_file_name = 'train.log'

    logfile = os.path.join(args.output_dir, log_file_name)
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))
    # setup device and distributed training
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device('cuda:{}'.format(args.local_rank))

        # initialize
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        # compute total world size (used to keep track of global step)
        args.world_size = int(os.environ['WORLD_SIZE'])  # torch.distributed.launch sets this to nproc_per_node * nnodes
    else:
        if torch.cuda.is_available(): args.local_rank = 0
        args.device = torch.device('cuda:{}'.format(args.local_rank) if args.local_rank is not None else 'cpu')

    # write ops only when on_main_process
    # NOTE: local_rank unique only to the machine; only 1 process on each node is on_main_process;
    #       if shared file system, args.local_rank below should be replaced by global rank e.g. torch.distributed.get_rank()
    args.on_main_process = (args.distributed and args.local_rank == 0) or not args.distributed

    # setup seed
    if args.seed:
        torch.manual_seed(args.seed)
        if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # load data; sets args.input_dims needed for setting up the model
    train_dataloader = fetch_dataloader(args, train=True)
    test_dataloader = fetch_dataloader(args, train=False)

    # load model
    model = Glow(args.width, args.depth, args.n_levels, args.input_dims, args.checkpoint_grads,lu_factorize=False).to(args.device)
    if args.distributed:
        # NOTE: DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        # for compatibility of saving/loading models, wrap non-distributed cpu/gpu model as well;
        # ie state dict is based on model.module.layer keys, which now match between training distributed and running then locally
        model = torch.nn.parallel.DataParallel(model)
    # DataParalle and DistributedDataParallel are wrappers around the model; expose functions of the model directly
    model.base_dist = model.module.base_dist
    model.log_prob = model.module.log_prob
    model.inverse = model.module.inverse

    # load optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load checkpoint if provided
    if args.restore_file:
        model_checkpoint = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location=args.device))
        args.start_epoch = model_checkpoint['epoch']
        args.step = model_checkpoint['global_step']

    # setup writer and outputs
    if args.on_main_process:
        writer = SummaryWriter(log_dir = args.output_dir)

        # save settings
        config = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__)) + \
                 'Num trainable params: {:,.0f}\n\n'.format(sum(p.numel() for p in model.parameters())) + \
                 'Model:\n{}'.format(model)
        config_path = os.path.join(args.output_dir, 'config.txt')
        writer.add_text('model_config', config)
        if not os.path.exists(config_path):
            with open(config_path, 'a') as f:
                print(config, file=f)

    if args.train:
        # run data dependent init and train
        data_dependent_init(model, args)
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, writer, args)

    if args.evaluate:
        logprob_mean, logprob_std = evaluate(model, test_dataloader, args)
        print('Evaluate: bits_x = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std))

    if args.generate:
        n_samples = 6
        z_std = [0., 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if not args.z_std else n_samples * [args.z_std]
        samples = generate(model, n_samples, z_std)
        images = make_grid(samples.cpu(), nrow=n_samples, pad_value=1)
        save_image(images, os.path.join(args.output_dir,
                                        'generated_samples_at_z_std_{}.png'.format('range' if args.z_std is None else args.z_std)))

    if args.visualize:
        if not args.z_std: args.z_std = 0.6
        if not args.vis_alphas: args.vis_alphas = [-2,-1,0,1,2]
        dec_x = visualize(model, args, args.vis_attrs, args.vis_alphas, args.vis_img)   # output (n_attr, n_alpha, 3, H, W)
        filename = 'manipulated_sample' if not args.vis_img else \
                   'manipulated_img_{}'.format(os.path.basename(args.vis_img).split('.')[0])
        if args.vis_attrs:
            filename += '_attr_' + ','.join(map(str, args.vis_attrs))
        save_image(dec_x.view(-1, *args.input_dims), os.path.join(args.output_dir, filename + '.png'), nrow=dec_x.shape[1])

    if args.on_main_process:
        writer.close()
