import scipy.misc
import os
import scipy.fftpack
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import copy

def hard_thresh(x, threshold):
    # hard threshold a tensor
    y = x.clone()
    y[y.abs() < threshold] = 0

    # # hard threshold a tensor
    # # y = x.clone()
    # x[x.abs() < threshold] = 0

    return y

def DCT(n):
    """
    #function to produce the DCT matrix
    """
    phi = scipy.fftpack.dct(np.eye(n), norm='ortho').T
    DCT = np.kron(phi, phi)
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
    target_image = np.divide(((image - image_min) * new_range), old_range) + target_min
    return target_image

# encoder performs transform
class deep_W_encoder(nn.Module):
    def __init__(self):
        super(deep_W_encoder, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # define the forward function
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.ReLU(self.fc3(x))
        x = self.ReLU(self.fc4(x))
        return x

# decoder performs inverse transform
class deep_W_decoder(nn.Module):
    def __init__(self):
        super(deep_W_decoder, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # define the forward function
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.ReLU(self.fc3(x))
        x = self.ReLU(self.fc4(x))
        return x

torch.cuda.set_device(0)

# Initialize networks
W_encoder = deep_W_encoder().cuda()
print(W_encoder)
W_decoder = deep_W_decoder().cuda()
print(W_decoder)

patch_size = 8
num_matches = 5
num_unmatches = 10
num_atoms = patch_size ** 2

threshold = 50
gamma_0 = 500               # Coefficient of log|det W|
gamma_1 = 0.5 * gamma_0     # Coefficient of the ||W||_F^2

img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barbara_512x512.npy").astype(float)
noisy_img = np.load(r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/barbara_512x512_noisy.npy").astype(float)

x_im = img                  # convert_range(img, np.amin(img), np.amax(img), -1, 1)
x_im_noisy = noisy_img      # convert_range(noisy_img, np.amin(noisy_img), np.amax(noisy_img), -1, 1)ape)

x = im2col(x_im, (patch_size, patch_size))
x_noisy = im2col(x_im_noisy, (patch_size, patch_size))
print('Shape after separating the image into columns of patch_size^2:', x.shape)
x = torch.tensor(x, dtype=torch.float).cuda()
x_noisy = torch.tensor(x_noisy, dtype=torch.float).cuda()
print('Shape after separating the image into columns of patch_size^2 in gpu:', x.shape)

learning_rate_encoder = 1e-5
learning_rate_decoder = 1e-5
num_steps = x.shape[1]
subset_steps = 10000
Nepoch = 1000
gap = 100

optimizer_encoder = torch.optim.Adam(W_encoder.parameters(), lr=learning_rate_encoder)
optimizer_decoder = torch.optim.Adam(W_decoder.parameters(), lr=learning_rate_decoder)
criterion = torch.nn.MSELoss(size_average=True)
cond = []
sparsity = []
epoch_losses = []
tx_losses = []
cyc_losses = []

match_inds = torch.zeros([x.shape[1], num_matches]).long().cuda()       # contains sorted matched indices for each reference patch
unmatch_inds = torch.zeros([x.shape[1], num_unmatches]).long().cuda()   # contains sorted unmatched indices for each reference patch
for epoch in range(584, Nepoch):
    tx_loss = 0
    cyc_loss = 0
    epoch_loss = 0
    epoch_fitloss = 0
    index_matrix = np.arange(0, num_steps, 1)                           # Initial reference patches
    index_subset = np.random.choice(index_matrix, size=subset_steps)

    # Update best matches for each reference patch by computing loss wrt all selected patches and finding out the minimum K ones
    Npatches = x.shape[1]
    for step in range(subset_steps):
        ref_ind = copy.copy(index_subset[step])
        x_ref_patch = x[:, ref_ind:ref_ind + 1]
        loss_patch_sorted = torch.argsort(torch.norm((x_ref_patch - x), dim=0))
        match_inds[ref_ind, :] = loss_patch_sorted[1:num_matches + 1]
        unmatch_inds[ref_ind, :] = loss_patch_sorted[
                                   1 + num_matches + gap:1 + num_matches + num_unmatches + gap]  # sort the norms, pick best matching K to K + D ones as unmatching

    # Optimize the filter W wrt selected K patches
    for step in range(subset_steps):  # batch size

        ref_ind = index_subset[step]

        x_ref = x[:, ref_ind:ref_ind + 1].view(1, x.shape[0])
        x_ref_noisy = x_noisy[:, ref_ind:ref_ind + 1].view(1, x_noisy.shape[0])
        x_matched = x_noisy[:, match_inds[ref_ind, :]].view(num_matches, x_noisy.shape[0])
        x_unmatched = x_noisy[:, unmatch_inds[ref_ind, :]].view(num_unmatches, x_noisy.shape[0])

        x_ref_noisy_tx = W_encoder(x_ref_noisy)
        x_matched_tx = W_encoder(x_matched)
        x_unmatched_tx = W_encoder(x_unmatched)

        x_ref1_cyc = W_decoder(hard_thresh(x_ref_noisy_tx.view(1, x_noisy.shape[0]), threshold=threshold))
        x_matched_cyc = W_decoder(hard_thresh(x_matched_tx.view(num_matches, x_noisy.shape[0]), threshold=threshold))
        x_unmatched_cyc = W_decoder(hard_thresh(x_unmatched_tx.view(num_unmatches, x_noisy.shape[0]), threshold=threshold))

        # Transform loss with not matching patches set
        loss_tx = torch.mean(torch.norm(hard_thresh(x_ref_noisy_tx, threshold) - hard_thresh(x_matched_tx, threshold), dim=1)) - \
                  torch.mean(torch.norm(hard_thresh(x_ref_noisy_tx, threshold) - hard_thresh(x_unmatched_tx, threshold), dim=1))

        # Transform loss without not matching patches set
        # loss_tx = torch.mean(torch.norm(hard_thresh(x_ref1_tx,threshold) - hard_thresh(x_matched_tx,threshold), dim=1))

        # Cyclic loss including only reference patch, can include all inverse transformed patches also
        loss_cyc = criterion(x_ref1_cyc, x_ref) #+ criterion(x_matched_cyc, x_matched) + criterion(x_unmatched_cyc, x_unmatched)
        # Sum both losses
        loss = loss_tx + loss_cyc

        # Update networks simultaneously (can be done separately to update encoder or decoder more frequently)
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()
        tx_loss += loss_tx.item()
        cyc_loss += loss_cyc.item()
        epoch_loss += loss.item()

    epoch_losses.append(epoch_loss)
    tx_losses.append(tx_loss)
    cyc_losses.append(cyc_loss)
    print('epoch', epoch)
    print('epoch_loss', epoch_loss/(subset_steps))
    print('cyc loss', cyc_loss/(subset_steps))
    print('tx loss', tx_loss/(subset_steps))

##############
# SAVE MODEL #
##############
path = r"/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM_results/results/deep_W_learning_cyclic/"
torch.save(
    {'epoch': epoch, 'model_state_dict': W_encoder.state_dict(), 'optimizer_state_dict': optimizer_encoder.state_dict(),
     'loss': loss},
    os.path.join(path,
                 'deep_cyclic_W_encoder_supervised_w_matched_%d_unmatched_%d_threshold_%.2f_gamma0_%.2f_refpatches_%d_subset_%d_BARBARA_512x512.tar' % (
                 num_matches, num_unmatches, threshold, gamma_0, num_steps, subset_steps)))
torch.save(
    {'epoch': epoch, 'model_state_dict': W_decoder.state_dict(), 'optimizer_state_dict': optimizer_decoder.state_dict(),
     'loss': loss},
    os.path.join(path,
                 'deep_cyclic_W_decoder_supervised_w_matched_%d_unmatched_%d_threshold_%.2f_gamma0_%.2f_refpatches_%d_subset_%d_BARBARA_512x512.tar' % (
                 num_matches, num_unmatches, threshold, gamma_0, num_steps, subset_steps)))

# # Check matched unmatched patches
# x_matched_patches = np.reshape(x_matched.cpu().detach().numpy(),[8,8,5])
# x_unmatched_patches = np.reshape(x_unmatched.cpu().detach().numpy(),[8,8,5])
# x_ref_patch = np.reshape(x_ref1.cpu().detach().numpy(), [8,8])
#
# xw_matched_patches = np.reshape(W@x_matched.cpu().detach().numpy(),[8,8,5])
# xw_unmatched_patches = np.reshape(W@x_unmatched.cpu().detach().numpy(),[8,8,5])
# xw_ref_patch = np.reshape(W@x_ref1.cpu().detach().numpy(), [8,8])
#
# plt.figure()
# plt.title('Reference patch')
# plt.imshow(np.reshape(x_ref1.cpu().detach().numpy(),[8,8]))
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.title('Reference patch transformed')
# plt.imshow(xw_ref_patch)
# plt.colorbar()
# plt.show()
#
# xw_ref_patch_ht = xw_ref_patch.copy()
# xw_ref_patch_ht[xw_ref_patch<threshold] = 0
# plt.figure()
# plt.title('Reference patch transformed HT')
# plt.imshow(xw_ref_patch_ht)
# plt.colorbar()
# plt.show()
#
# fig,axs = plt.subplots(1,5)
# plt.suptitle('transformed matched patches')
# im0 = axs[0].imshow(xw_matched_patches[:,:,0])
# im1 = axs[1].imshow(xw_matched_patches[:,:,1])
# im2 = axs[2].imshow(xw_matched_patches[:,:,2])
# im3 = axs[3].imshow(xw_matched_patches[:,:,3])
# im4 = axs[4].imshow(xw_matched_patches[:,:,4])
# plt.show()
#
# fig,axs = plt.subplots(1,5)
# plt.suptitle('transformed non-matching patches')
# im0 = axs[0].imshow(xw_unmatched_patches[:,:,0])
# im1 = axs[1].imshow(xw_unmatched_patches[:,:,1])
# im2 = axs[2].imshow(xw_unmatched_patches[:,:,2])
# im3 = axs[3].imshow(xw_unmatched_patches[:,:,3])
# im4 = axs[4].imshow(xw_unmatched_patches[:,:,4])
# plt.show()
#
# plt.figure()
# plt.imshow(hard_thresh(xw_ref_patch,threshold=100))
# plt.show()
#
# def hard_thresh(x,threshold):
#     # hard threshold a tensor
#     y = x.copy()
#     y[np.abs(y) < threshold] = 0
#     return y
#
# fig,axs = plt.subplots(1,5)
# plt.suptitle('transformed HT matching patches')
# im0 = axs[0].imshow(hard_thresh(xw_matched_patches[:,:,0],threshold=100),clim=(0,200))
# im1 = axs[1].imshow(hard_thresh(xw_matched_patches[:,:,1],threshold=100),clim=(0,200))
# im2 = axs[2].imshow(hard_thresh(xw_matched_patches[:,:,2],threshold=100),clim=(0,200))
# im3 = axs[3].imshow(hard_thresh(xw_matched_patches[:,:,3],threshold=100),clim=(0,200))
# im4 = axs[4].imshow(hard_thresh(xw_matched_patches[:,:,4],threshold=100),clim=(0,200))
# plt.show()
#
# fig,axs = plt.subplots(1,5)
# plt.suptitle('transformed HT non-matching patches')
# im0 = axs[0].imshow(hard_thresh(xw_unmatched_patches[:,:,0],threshold=100),clim=(0,200))
# im0 = axs[1].imshow(hard_thresh(xw_unmatched_patches[:,:,1],threshold=100),clim=(0,200))
# im0 = axs[2].imshow(hard_thresh(xw_unmatched_patches[:,:,2],threshold=100),clim=(0,200))
# im0 = axs[3].imshow(hard_thresh(xw_unmatched_patches[:,:,3],threshold=100),clim=(0,200))
# im0 = axs[4].imshow(hard_thresh(xw_unmatched_patches[:,:,4],threshold=100),clim=(0,200))
# plt.show()