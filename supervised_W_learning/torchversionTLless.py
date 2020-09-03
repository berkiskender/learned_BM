import scipy.misc
import os
import scipy.fftpack
import numpy as np
import torch
import matplotlib.pyplot as plt
# import line_profiler
# import cv2
import imageio



def convert_range(image, image_min, image_max, target_min, target_max):
    """
    :return reshape the image to range between 1 and -1:
    """
    old_range = image_max - image_min
    new_range = target_max - target_min
    target_image = np.divide(((image-image_min) * new_range),old_range) + target_min
    return target_image

plt.set_cmap('gray')

patch_size = 8

num_matches = 5 # 20
num_unmatches =  10
num_atoms = patch_size ** 2

threshold = 0.1
#0.9 good TF, 0.15,0.09 0.28
gamma_0 = 500 #10  good TF  20 300
print(gamma_0)
gamma_1 = 0.5*gamma_0 # 0.1 good TF  10

learning_rate = 5e-2 # 5e-2
num_steps = 1000 # 15

img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barabrareshape.npy")
img = img.astype(float)
noisy_img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barabranoise3.npy")
noisy_img = noisy_img.astype(float)
x_im = convert_range(img, np.amin(img), np.amax(img), -1, 1)
x_im_noisy = convert_range(noisy_img, np.amin(noisy_img), np.amax(noisy_img), -1, 1)

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

x = im2col(x_im, (patch_size, patch_size))
print(x.shape)
x_noisy = im2col(x_im_noisy, (patch_size, patch_size))
x = torch.tensor(x, dtype=torch.float)
x_noisy = torch.tensor(x_noisy, dtype=torch.float)
print(x.shape)

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

W_DCT = DCT(8)
W = DCT(8)
W = torch.tensor(W).float()
W.requires_grad_(True)
opti = torch.optim.Adam([W], lr=learning_rate)
#loss = []
#fitloss = []
cond = []
sparsity = []
epoch_losses = []
epoch_fitlosses =[]
# precompute the correct index first and save it as matrix
for epoch in range(200):
    epoch_loss = 0
    epoch_fitloss = 0
    for step in range(num_steps):# batch size
        ref_ind = step#np.random.randint(x.shape[1])# pick random ref patch
        x_ref = x[:, ref_ind:ref_ind + 1]
        norms = np.linalg.norm((x_ref-x), axis=0)# norm matrix
        match_inds = np.argsort(norms)[1:num_matches + 1]
        un_match_inds = np.argsort(norms)[1+num_matches:1+num_matches+num_unmatches]
        x_ref1 = x_noisy[:, ref_ind:ref_ind + 1]
        x_matched = x_noisy[:, match_inds]
        x_unmatched = x_noisy[:,un_match_inds]
        loss_fit = torch.mean(torch.norm(hard_thresh(W @ x_ref1,threshold) - hard_thresh(W @ x_matched,threshold), dim=0))\
                   -torch.mean(torch.norm(hard_thresh(W @ x_ref1,threshold) - hard_thresh(W @ x_unmatched,threshold), dim=0))
        loss_reg = - gamma_0 * W.slogdet()[1] + gamma_1 * torch.sum(torch.abs(W)**2) #torch.norm(W)#torch.sum((W)**2)
        loss = loss_fit + loss_reg
        epoch_loss += loss.item()
        epoch_fitloss += loss_fit.item()

        W1 = W.detach().numpy()
        cond.append(np.linalg.cond(W1))

        opti.zero_grad()
        loss.backward()
        opti.step()
        P2block = hard_thresh(W@x_matched,threshold).detach().numpy()
        sparsity.append((np.count_nonzero(P2block) / np.prod(np.shape(P2block))) * 100)
    epoch_losses.append(epoch_loss)
    epoch_fitlosses.append(epoch_fitloss)
    print('epoch',epoch)
    print('epoch_loss',epoch_loss)
    print('epoch_fitloss',epoch_fitloss)
W = W.detach().numpy()
path = r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/results"
np.save(os.path.join(path,'Wmatrix.npy'),W)
#57,48, 58 is good matrix
#113 have great accuracy but not good pattern

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