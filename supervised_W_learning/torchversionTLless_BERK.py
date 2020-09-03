import scipy.misc
import os
import scipy.fftpack
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

# import line_profiler
# import cv2
import imageio

import time

torch.cuda.set_device(0)

def im2col(A, BSZ, stepsize=1):
    # reshape image to column that help for searching simliar block
    # go back with X[:, ind].reshape(patch_size, patch_size))
    m, n = A.shape
    s0, s1 = A.strides
    nrows = m - BSZ[0] + 1
    ncols = n - BSZ[1] + 1
    shp = BSZ[0], BSZ[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0] * BSZ[1], -1)[:, ::stepsize]

def hard_thresh(x,threshold):
    # hard threshold a tensor
    y = x.clone().cuda()
    y[y.abs() < threshold] = 0
    return y

def DCT(n):
    """
    #function to produce the DCT matrix
    """
    phi = scipy.fftpack.dct(np.eye(n), norm='ortho').T
    DCT = np.kron(phi,phi)
    return DCT

def convert_range(image, image_min, image_max, target_min, target_max):
    """
    :return reshape the image to range between 1 and -1:
    """
    old_range = image_max - image_min
    new_range = target_max - target_min
    target_image = np.divide(((image-image_min) * new_range),old_range) + target_min
    return target_image


patch_size = 8
num_matches = 5 # 20
num_unmatches =  10
num_atoms = patch_size ** 2

threshold = 0.5
#0.9 good TF, 0.15,0.09 0.28
gamma_0 = 1000 #10  good TF  20 300
print(gamma_0)
gamma_1 = 0.5*gamma_0 # 0.1 good TF  10

learning_rate = 5e-2 # 5e-2
num_steps = 1000 # 15

# Load images
img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barabrareshape.npy").astype(float)
noisy_img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barabranoise3.npy").astype(float)
x_im = convert_range(img, np.amin(img), np.amax(img), -1, 1)
x_im_noisy = convert_range(noisy_img, np.amin(noisy_img), np.amax(noisy_img), -1, 1)

# Convert images to patchified forms
x = im2col(x_im, (patch_size, patch_size))
x_noisy = im2col(x_im_noisy, (patch_size, patch_size))
x = torch.tensor(x, dtype=torch.float).cuda()
x_noisy = torch.tensor(x_noisy, dtype=torch.float).cuda()
print(x.shape)

# Initialize W as the DCT matrix
W = DCT(8)
W = torch.tensor(W).float().cuda()
W.requires_grad_(True)


opti = torch.optim.Adam([W], lr=learning_rate)
cond = []
sparsity = []
epoch_losses = []
epoch_fitlosses =[]
# precompute the correct index first and save it as matrix
index_matrix = np.arange(0, num_steps, 1)
for epoch in range(200,400):
    epoch_loss = 0
    epoch_fitloss = 0
    # BERK: train with random ref_ind order
    np.random.shuffle(index_matrix)
    # loss = 0
    for step in range(num_steps):# batch size
        ref_ind = index_matrix[step]
        x_ref = x[:, ref_ind:ref_ind + 1]
        norms = torch.norm((x_ref-x), dim=0) # norm matrix

        sorted_norms = torch.argsort(norms)
        match_inds = sorted_norms[1:num_matches + 1] # sort the norms, pick best matching K ones except for the patch itself
        un_match_inds = sorted_norms[1+num_matches:1+num_matches+num_unmatches] # sort the norms, pick best matching K to K + D ones as unmatching

        x_ref1 = x_noisy[:, ref_ind:ref_ind + 1] # Pick reference patch from noisy image
        x_matched = x_noisy[:, match_inds] # Pick K matched patches from the noisy image
        x_unmatched = x_noisy[:,un_match_inds] # Pick D unmatching patches from the noisy image

        loss_fit = torch.mean(torch.norm(hard_thresh(torch.mm(W, x_ref1),threshold) - hard_thresh(torch.mm(W, x_matched),threshold), dim=0))\
                   - torch.mean(torch.norm(hard_thresh(torch.mm(W, x_ref1),threshold) - hard_thresh(torch.mm(W, x_unmatched),threshold), dim=0))

        loss_reg = - gamma_0 * W.slogdet()[1] + gamma_1 * torch.sum(torch.abs(W)**2) #torch.norm(W)#torch.sum((W)**2)

        loss = loss_fit + loss_reg
        epoch_loss += loss.item()
        epoch_fitloss += loss_fit.item()

        # loss += loss_fit + loss_reg
        # epoch_loss += (loss_fit + loss_reg).item()
        # epoch_fitloss += loss_fit.item()

        opti.zero_grad()
        loss.backward()
        opti.step()

    epoch_losses.append(epoch_loss)
    epoch_fitlosses.append(epoch_fitloss)
    print('epoch',epoch)
    print('epoch_loss',epoch_loss)
    print('epoch_fitloss',epoch_fitloss)

    start_rest = time.time()
    W1 = W.cpu().detach().numpy()
    cond.append(np.linalg.cond(W1))
    P2block = hard_thresh(W @ x_matched, threshold).cpu().detach().numpy()
    sparsity.append((np.count_nonzero(P2block) / np.prod(np.shape(P2block))) * 100)
    end_rest = time.time()
    rest_time = end_rest - start_rest

W = W.cpu().detach().numpy()
path = r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/results"
np.save(os.path.join(path,'Wmatrix_on_gpu.npy'),W)
#57,48, 58 is good matrix
#113 have great accuracy but not good pattern


# plt.set_cmap('gray')
fig = plt.figure()
fig.add_subplot(221)
plt.title('Loss')
plt.plot(epoch_losses)
fig.add_subplot(222)
plt.title('fitloss')
plt.plot(epoch_fitlosses)
fig.add_subplot(223)
plt.title('cond')
plt.plot(cond)
fig.add_subplot(224)
plt.title('sparsity')
plt.plot(sparsity)
plt.show()
