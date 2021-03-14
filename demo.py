# coding: utf-8
# Copyright (c) Shuai Liao. All Rights Reserved.

"""
Disclaimer:
    This code is simply a pytorch translation of:
    https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
"""

import torch
import torch.nn as nn
import imageio
import numpy as np

from pylibs.common import plt, plt_save, generate_animation, cv2_putText  # cv2_imshow, cv2_wait, cv2_putText,
from pylibs.reducer_v2 import  reducer_group
from tqdm.notebook import tqdm as tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load image
image_path = 'assets/fox.jpg'

img = imageio.imread(image_path).astype(np.float32) / 255.   # image shape: (512, 512, 3)



# Create input pixel coordinates in the unit square
coords = np.linspace(0, 1, img.shape[0], endpoint=False, dtype=np.float32)
x_test = np.stack(np.meshgrid(coords, coords), -1)
test_data  = [x_test, img]                      # shapes: (512, 512, 2), (512, 512, 3)
train_data = [x_test[::2,::2], img[::2,::2]]    # shapes: (256, 256, 2), (256, 256, 3)
#
test_data  = list(map(lambda x: torch.tensor(x).to(device), test_data) )
train_data = list(map(lambda x: torch.tensor(x).to(device), train_data))



# Fourier feature mapping
def input_mapping(x, B):
    if B is None:
        return x
    else:
        B = B.to(x.device)
        x_proj = (2.*np.pi*x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)



def make_network(num_layers, num_channels, input_channels=2):
    layers = [ nn.Linear(input_channels, num_channels) ]
    for i in range(num_layers-1):
        layers.append(nn.Linear(num_channels, num_channels))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(num_channels, 3))
    layers.append(nn.Sigmoid())
    net = nn.Sequential(*layers)

    return net


def compute_loss(net, B, input, target):
    input = input_mapping(input, B)
    h,w,ch = input.shape
    input, target = input.view(h*w,ch), target.view(h*w,3)

    pred = net(input)
    loss = 0.5 * torch.mean((pred - target)**2)
    return loss, pred.view(h,w,3)



# Train model with given hyperparameters and data
def train_model(network_size, learning_rate, iters, B, train_data, test_data):

    net = make_network(*network_size, input_channels=2 if B is None else len(B)*2)
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    g_rd = reducer_group(['train_psnr', 'test_psnr', 'pred_img', 'step'])  # PSNR: peak signal-to-noise ratio
    for i in tqdm(range(iters), desc='train iter', leave=False):

        loss_train, pred_train = compute_loss(net, B, *train_data)

        if (i+1) % 25 == 0:
            with torch.no_grad():
                loss_test, pred_test = compute_loss(net, B, *test_data)

                print(f'step={i+1:4d}  loss_train={loss_train.item():.3f}')
                pred_img = pred_test.data.cpu().numpy()
                cv2_putText(pred_img, (5,40), f'#{i+1} loss:{loss_train.item():.3f}', scale=1.5, fgcolor=(0,0,1.), thickness=2,)
            #
            g_rd.collect(dict(
                train_psnr = -10 * torch.log10(2.*loss_train),
                test_psnr  = -10 * torch.log10(2.*loss_test),
                pred_img   = pred_img[None,:,:,:].clip(0,1),
                step       = i,
                ), squeeze=False)

        optimizer.zero_grad()   # clear gradients for next train
        loss_train.backward()   # backpropagation, compute gradients
        optimizer.step()        # apply gradients


    # train_psnrs, test_psnrs, pred_imgs, xs = g_rd.reduce().values()
    return g_rd.reduce()




network_size = (4, 256)
learning_rate = 1e-4
iters = 2000

mapping_size = 256


B_dict = {}
# Standard network - no mapping
B_dict['none'] = None
# Basic mapping
B_dict['basic'] = torch.eye(2)
# Three different scales of Gaussian Fourier feature mappings
B_gauss = torch.normal(0,1,size=(mapping_size,2))
for scale in [1., 10., 100.]:
    B_dict[f'gauss_{scale}'] = B_gauss * scale



# This should take about 2-3 minutes
outputs = {}
for k in tqdm(B_dict):
    outputs[k] = train_model(network_size, learning_rate, iters, B_dict[k], train_data, test_data)
    generate_animation(f'assets/output.cache/{k}.gif', outputs[k]['pred_img'], rsz_height=256, duration=0.2)
    # generate_mp4(f'assets/{k}.mp4', outputs[k]['pred_img'], rsz_height=256, duration=1)



# Plot train/test error curves
plt.figure(figsize=(16,6))

plt.subplot(121)
for i, k in enumerate(outputs):
    plt.plot(outputs[k]['step'], outputs[k]['train_psnr'], label=k)
plt.title('Train error', fontsize=18)
plt.ylabel('PSNR', fontsize=18)
plt.xlabel('Training iter', fontsize=18)
plt.legend()

plt.subplot(122)
for i, k in enumerate(outputs):
    plt.plot(outputs[k]['step'], outputs[k]['test_psnr'], label=k)
plt.title('Test error', fontsize=18)
plt.ylabel('PSNR', fontsize=18)
plt.xlabel('Training iter', fontsize=18)
plt.legend()

# plt.show()
plt_save('assets/output.cache/training_curve.png')
