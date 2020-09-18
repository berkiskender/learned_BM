import scipy.misc
import os
import scipy.fftpack
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
# import line_profiler
# import cv2
# import imageio
import copy

def hard_thresh(x,threshold):
    # hard threshold a tensor
    y = x.clone()
    y[y.abs() < threshold] = 0
    return y

def DCT(n):
    """
    #function to produce the DCT matrix
    """
    phi = scipy.fftpack.dct(np.eye(n), norm='ortho').T
    DCT = np.kron(phi,phi)
    return DCT

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

def convert_range(image, image_min, image_max, target_min, target_max):
    """
    :return reshape the image to range between 1 and -1:
    """
    old_range = image_max - image_min
    new_range = target_max - target_min
    target_image = np.divide(((image-image_min) * new_range),old_range) + target_min
    return target_image

# for param_group in opti.param_groups:
#     param_group['lr'] = learning_rate

torch.cuda.set_device(0)

patch_size = 8
num_matches = 5 # 20
num_unmatches = 10
num_atoms = patch_size ** 2

threshold = 80
gamma_0 = 500           # Coefficient of log|det W|
gamma_1 = 0.5*gamma_0   # Coefficient of the ||W||_F^2

img = np.load(os.path.join('..','BM3D','barbara_512x512.npy')).astype(float)
noisy_img = np.load(os.path.join('..','BM3D','barbara_512x512_noisy.npy')).astype(float)

x_im = img #convert_range(img, np.amin(img), np.amax(img), -1, 1)
x_im_noisy = noisy_img #convert_range(noisy_img, np.amin(noisy_img), np.amax(noisy_img), -1, 1)ape)

x = im2col(x_im, (patch_size, patch_size))
x_noisy = im2col(x_im_noisy, (patch_size, patch_size))
print('Shape after separating the image into columns of patch_size^2:',x.shape)
x = torch.tensor(x, dtype=torch.float).cuda()
x_noisy = torch.tensor(x_noisy, dtype=torch.float).cuda()
print('Shape after separating the image into columns of patch_size^2 in gpu:',x.shape)

# Initialize the filter with DCT
W1 = DCT(patch_size)
# Initialize the filter with identity
W2 = np.eye(8**2)

W3 = np.eye(8**2)
W4 = np.eye(8**2)
W1 = torch.tensor(W1).float().cuda()
W1.requires_grad_(True)

W2 = torch.tensor(W2).float().cuda()
W2.requires_grad_(True)

W3 = torch.tensor(W3).float().cuda()
W3.requires_grad_(True)
W4 = torch.tensor(W4).float().cuda()
W4.requires_grad_(True)
learning_rate = 1e-5
num_steps = x.shape[1] # 15
subset_steps = 10000
Nepoch = 1000
gap = 100

optimizer_W1 = torch.optim.Adam([W1], lr=learning_rate)
optimizer_W2 = torch.optim.Adam([W2], lr=learning_rate)
optimizer_W3 = torch.optim.Adam([W3], lr = learning_rate)
optimizer_W4 = torch.optim.Adam([W4], lr = learning_rate)
cond = []
sparsity = []
epoch_losses = []
epoch_fitlosses =[]

match_inds = torch.zeros([x.shape[1],num_matches]).long().cuda() # contains sorted matched indices for each reference patch
unmatch_inds = torch.zeros([x.shape[1],num_unmatches]).long().cuda() # contains sorted unmatched indices for each reference patch
for epoch in range(0,Nepoch):
    epoch_loss = 0
    epoch_fitloss = 0
    index_matrix = np.arange(0,num_steps,1) # Initial reference patches
    # np.random.shuffle(index_matrix) # shuffle the training order for W for selection of reference patches
    index_subset = np.random.choice(index_matrix, size = subset_steps)

    # Update best matches for each reference patch by computing loss wrt all selected patches and finding out the minimum K ones
    # Initialize with DCT
    Npatches = x.shape[1]
    # prev_ind = copy.copy(num_steps)
    for step in range(subset_steps):
        ref_ind = copy.copy(index_subset[step])
        x_ref1 = x[:, ref_ind:ref_ind+1]
        loss_patch_sorted = torch.argsort(torch.norm((x_ref1-x), dim=0))
        match_inds[ref_ind,:] = loss_patch_sorted[1:num_matches+1]
        unmatch_inds[ref_ind,:] = loss_patch_sorted[1+num_matches+gap:1+num_matches+num_unmatches+gap] # sort the norms, pick best matching K to K + D ones as unmatching

    # Optimize the filter W wrt selected K patches
    # loss = 0
    for step in range(subset_steps): # batch size

        ref_ind = index_subset[step]
        x_ref1 = W1@(F.relu(W2@(F.relu(W3@(F.relu(W4@x_noisy[:, ref_ind:ref_ind+1]))))))
        x_matched = W1@(F.relu(W2@(F.relu(W3@(F.relu(W4@x_noisy[:, match_inds[ref_ind,:]]))))))
        x_unmatched = W1@(F.relu(W2@(F.relu(W3@(F.relu(W4@x_noisy[:, unmatch_inds[ref_ind,:]]))))))
        loss_fit = torch.mean(torch.norm(hard_thresh(x_ref1,threshold) - hard_thresh(x_matched,threshold), dim=0)) - \
            torch.mean(torch.norm(hard_thresh(x_ref1,threshold) - hard_thresh(x_unmatched,threshold), dim=0))
        loss_reg = - gamma_0 * W1.slogdet()[1] + gamma_1 * torch.sum(torch.abs(W1)**2) - gamma_0 * W2.slogdet()[1] + gamma_1 * torch.sum(torch.abs(W2)**2)-gamma_0 *W3.slogdet()[1] + gamma_1*torch.sum(torch.abs(W3)**2)-gamma_0*W4.slogdet()[1] + gamma_1*torch.sum(torch.abs(W4)**2)

        loss = loss_fit + loss_reg

        optimizer_W1.zero_grad()
        optimizer_W2.zero_grad()
        optimizer_W3.zero_grad()
        optimizer_W4.zero_grad()
        loss.backward()
        optimizer_W1.step()
        optimizer_W2.step()
        optimizer_W3.step()
        optimizer_W4.step()
        epoch_loss += loss.item()
        epoch_fitloss += loss_fit.item()

    epoch_losses.append(epoch_loss)
    epoch_fitlosses.append(epoch_fitloss)
    print('epoch',epoch)
    print('epoch_loss',epoch_loss)
    print('epoch_fitloss',epoch_fitloss)
W1 = W1.cpu().detach().numpy()
#path = r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/results/multi_W_learning"
#np.save(os.path.join(path,'BARBARA_512x512_LATEST.npy' %(num_matches,num_unmatches,threshold,gamma_0,num_steps, subset_steps,gap)),W1)
W2 = W2.cpu().detach().numpy()
#path = r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/results/multi_W_learning"
#np.save(os.path.join(path,'W2_supervise.npy' %(num_matches,num_unmatches,threshold,gamma_0,num_steps, subset_steps,gap)),W2)
W3 = W3.cpu().detach().numpy()
