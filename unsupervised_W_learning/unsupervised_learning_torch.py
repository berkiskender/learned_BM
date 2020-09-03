import scipy
import scipy.misc
import os
import scipy.fftpack
import numpy as np
import torch
import matplotlib.pyplot as plt
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

torch.cuda.set_device(0)

patch_size = 8
num_matches = 5 # 20
num_unmatches = 5
gap = 20
num_atoms = patch_size ** 2

threshold = 100
gamma_0 = 500           # Coefficient of log|det W|
gamma_1 = 0.5*gamma_0   # Coefficient of the ||W||_F^2

learning_rate = 5e-4
subset_steps = 10000
Nepoch = 1000

# img = imageio.imread('/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barbara_512x512.png').astype(np.float64)
# # img = img[::2,::2]
# # img = (0.3 * img[:,:,0]) + (0.59 * img[:,:,1]) + (0.11 * img[:,:,2])
# np.save('/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barbara_512x512', img)
# noisy_img = img + np.random.normal(loc = 0, scale = 30, size=img.shape)
# np.save('/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barbara_512x512_noisy', noisy_img)

# # Barbara img
# img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barabrareshape.npy").astype(float)
# noisy_img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barabranoise3.npy").astype(float)

# # Boat img
# img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/boat.npy").astype(float)
# noisy_img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/boat_noisy.npy").astype(float)

# # Airplane img
# img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/airplane_bw.npy").astype(float)
# noisy_img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/airplane_bw_noisy.npy").astype(float)

# Barbara img 512x512
img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barbara_512x512.npy").astype(float)
noisy_img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barbara_512x512_noisy.npy").astype(float)

# x_im = convert_range(img, np.amin(img), np.amax(img), -1, 1)
# x_im_noisy = convert_range(noisy_img, np.amin(noisy_img), np.amax(noisy_img), -1, 1)
x_im = img
x_im_noisy = noisy_img

# Gaussian filtering for initial assignments
from scipy.ndimage import gaussian_filter
x_im_gauss_denoised = gaussian_filter(x_im_noisy, sigma=1)
plt.figure()
plt.imshow(x_im_gauss_denoised, cmap='gray')
plt.show()

x = im2col(x_im, (patch_size, patch_size))
x_noisy = im2col(x_im_noisy, (patch_size, patch_size))
x_denoised = im2col(x_im_gauss_denoised, (patch_size, patch_size))
print('Shape after separating the image into columns of patch_size^2:',x.shape)
x = torch.tensor(x, dtype=torch.float).cuda()
x_noisy = torch.tensor(x_noisy, dtype=torch.float).cuda()
x_denoised = torch.tensor(x_denoised, dtype=torch.float).cuda()
print('Shape after separating the image into columns of patch_size^2 in gpu:',x.shape)

# Initialize the filter with DCT
W = DCT(patch_size)
# # Initialize the filter with identity
# W = np.eye(8**2)
# # Initialize the filter randomly
# W = np.random.normal(0,1,size=[64,64])

W = torch.tensor(W).float().cuda()
W.requires_grad_(True)

opti = torch.optim.Adam([W], lr=learning_rate)
cond = []
sparsity = []
epoch_losses = []
epoch_fitlosses =[]

num_steps = x.shape[1] # 15
match_inds = torch.zeros([x.shape[1],num_matches]).long().cuda() # contains sorted matched indices for each reference patch
unmatch_inds = torch.zeros([x.shape[1],num_unmatches]).long().cuda() # contains sorted unmatched indices for each reference patch
index_matrix = np.arange(0, num_steps, 1)  # Initial reference patches

# Initial block matching wrt a smoothed noisy image
print('Initial block matching w smoothed img:')
for step in range(subset_steps):
    index_subset = np.random.choice(index_matrix, size=subset_steps)
    ref_ind = copy.copy(index_subset[step])
    x_ref1 = x_denoised[:, ref_ind:ref_ind + 1]
    loss_patch_sorted = torch.argsort(
        torch.norm(hard_thresh(W @ x_ref1, threshold) - hard_thresh(W @ x_denoised, threshold), dim=0))
    match_inds[ref_ind, :] = loss_patch_sorted[1:num_matches + 1]
    unmatch_inds[ref_ind, :] = loss_patch_sorted[
                               1 + num_matches + gap:1 + num_matches + num_unmatches+ gap]  # sort the norms, pick best matching K to K + D ones as unmatching
print('Initial block matching w smoothed img done.')

for epoch in range(0,Nepoch):
    epoch_loss = 0
    epoch_fitloss = 0

    # Update best matches for each reference patch by computing loss wrt all selected patches and finding out the minimum K ones
    # Initialize with DCT
    Npatches = x.shape[1]
    if epoch > 10 and epoch % 2 == 0:
        index_subset = np.random.choice(index_matrix, size=subset_steps)
        for step in range(subset_steps):
            ref_ind = copy.copy(index_subset[step])
            x_ref1 = x_noisy[:, ref_ind:ref_ind+1]
            loss_patch_sorted = torch.argsort(torch.norm(hard_thresh(W @ x_ref1,threshold) - hard_thresh(W @ x_noisy,threshold), dim=0))
            match_inds[ref_ind,:] = loss_patch_sorted[1:num_matches+1]
            unmatch_inds[ref_ind,:] = loss_patch_sorted[1+num_matches:1+num_matches+num_unmatches] # sort the norms, pick best matching K to K + D ones as unmatching

    # Optimize the filter W wrt selected K patches
    # loss = 0
    for step in range(subset_steps): # batch size

        ref_ind = index_subset[step]
        x_ref1 = x_noisy[:, ref_ind:ref_ind+1]
        x_matched = x_noisy[:, match_inds[ref_ind,:]]
        x_unmatched = x_noisy[:, unmatch_inds[ref_ind,:]]
        loss_fit = torch.mean(torch.norm(hard_thresh(W @ x_ref1,threshold) - hard_thresh(W @ x_matched,threshold), dim=0)) - \
                   torch.mean(torch.norm(hard_thresh(W @ x_ref1,threshold) - hard_thresh(W @ x_unmatched,threshold), dim=0))
        loss_reg = - gamma_0 * W.slogdet()[1] + gamma_1 * torch.sum(torch.abs(W)**2)

        loss = loss_fit + loss_reg

        opti.zero_grad()
        loss.backward()
        opti.step()

        epoch_loss += loss.item()
        epoch_fitloss += loss_fit.item()

    # epoch_loss = loss.item()
    # epoch_fitloss = loss_fit.item()

    # W1 = W.detach().numpy()
    # cond.append(np.linalg.cond(W1))
    # P2block = hard_thresh(W@x_matched,threshold).detach().numpy()
    # sparsity.append((np.count_nonzero(P2block) / np.prod(np.shape(P2block))) * 100)

    epoch_losses.append(epoch_loss)
    epoch_fitlosses.append(epoch_fitloss)
    print('epoch',epoch)
    print('epoch_loss',epoch_loss)
    print('epoch_fitloss',epoch_fitloss)

W = W.cpu().detach().numpy()
x_noisy = x_noisy.cpu().detach().numpy()
path = r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/results"
np.save(os.path.join(path,'Wmatrix_unsupervised_w_matched_%d_unmatched_%d_DCT_init_threshold_%.2f_gamma0_%.2f_refpatches_%d_subset_%d_BARBARA_512x512_LATEST.npy' %(num_matches,num_unmatches,threshold,gamma_0,num_steps, subset_steps)),W)
#57,48, 58 is good matrix
#113 have great accuracy but not good pattern

# Kronecker product display
plt.figure()
plt.subplot(1,3,1)
plt.title('Learned W')
plt.imshow(W)
plt.colorbar()
plt.subplot(1,3,2)
plt.title('DCT')
plt.imshow(DCT(8))
plt.colorbar()
plt.subplot(1,3,3)
plt.title('Learned W - DCT')
plt.imshow(W-DCT(8))
plt.colorbar()
plt.show()

# Display Rows
W_rows = np.reshape(W, [64,8,8], order='F')
DCT_rows = np.reshape(DCT(8), [64,8,8], order='F')

fig, axs = plt.subplots(8,8,figsize=(15,15))
for i in range(0,8):
    for j in range(0, 8):
        axs[i, j].imshow(W_rows[8*i+j,:,:])
plt.show()

fig, axs = plt.subplots(8,8,)
for i in range(0,8):
    for j in range(0, 8):
        axs[i, j].imshow(DCT_rows[8*i+j,:,:])
plt.show()

ref_ind = 10000
for i in range(0,num_matches):
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(x_noisy[:,ref_ind],[8,8]),cmap='gray')
    plt.clim(0,255)
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(x_noisy[:, match_inds[ref_ind, i]],[8,8]),cmap='gray')
    plt.clim(0,255)
    plt.colorbar()
    plt.show()

torch.norm(hard_thresh(W @ x_noisy[:,ref_ind],threshold) - hard_thresh(W @ x_noisy[:, match_inds[ref_ind, i]],threshold), dim=0)

fig = plt.figure()
fig.add_subplot(121)
plt.title('Loss')
plt.plot(epoch_losses)
fig.add_subplot(122)
plt.title('fitloss')
plt.plot(epoch_fitlosses)
# fig.add_subplot(223)
# plt.title('cond')
# plt.plot(cond)
# fig.add_subplot(224)
# plt.title('sparsity')
# plt.plot(sparsity)
plt.show()