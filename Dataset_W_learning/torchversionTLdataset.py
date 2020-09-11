import scipy.misc
import os
import scipy.fftpack
import numpy as np
import torch
import matplotlib.pyplot as plt
#import line_profiler
import cv2
import glob
#def convert_range(image, image_min, image_max, target_min, target_max):
#    """
#    :return reshape the image to range between 1 and -1:
#    """
#    old_range = image_max - image_min
#    new_range = target_max - target_min
#    target_image = np.divide(((image-image_min) * new_range),old_range) + target_min
#    return target_image

#plt.set_cmap('gray')

patch_size = 8


num_matches = 5 # 20
num_unmatches =  20
num_atoms = patch_size ** 2

threshold = 80
#0.9 good TF, 0.15,0.09 0.28
gamma_0 = 50e1  #10  good TF  20 300 100e1
print(gamma_0)
gamma_1 = 0.5*gamma_0 # 0.1 good TF  10

learning_rate = 5e-4# 5e-2
batchs_size = 1000 # 15
num_image = 20
num_steps = 110

#x_im[0::2, :] += .5


#def im2col(mtx, block_size): # functinon change image to column
#    mtx_shape = mtx.shape
#    sx = mtx_shape[0] - block_size[0] + 1
#    sy = mtx_shape[1] - block_size[1] + 1

#    result = np.empty((block_size[0] * block_size[1], sx * sy))

#    for i in range(sy):
#        for j in range(sx):
#            result[:, i * sx + j] = mtx[i:i + block_size[0], j:j + block_size[1]].flatten()
#    return result
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
torch.cuda.set_device(0)
W = DCT(8)
W = torch.tensor(W).float().cuda()
W.requires_grad_(True)
opti = torch.optim.Adam([W], lr=learning_rate)
#loss = []
#fitloss = []
cond = []
sparsity = []
epoch_losses = []
epoch_fitlosses =[]

# precompute the correct index first and save it as matrix
# the loss_fit could use the indexing for the hardthreshd noisy image rather than multipy is several times
# by using the stochastic way to change the ref_ind randomly rather than whole image
files = glob.glob(os.path.join("..","residue_data","image_SRF_2_noise_sigma_50", '*.npy'))# it used to attach all the image from the folder
files.sort()
files2 = glob.glob(os.path.join("..","residue_data","image_SRF_2_reshape", '*.npy'))
files2.sort()
image_no = 1

image_list =[]
noisy_img_list = []
for j in range(num_image):
   # img = cv2.imread(files2[j])
   # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float)
    img = np.load(files2[j]).astype(np.float)
    noisy_img = np.load(files[j]).astype(np.float)
    image_list.append(img)
    noisy_img_list.append(noisy_img)
for step in range(num_steps):
    step_loss = 0
    step_fitloss = 0
    for i in range(num_image):
        img1 = image_list[i]
        noisy_img1 = noisy_img_list[i]
        x = im2col(img1, (patch_size, patch_size))
       # x_noisy = im2col(noisy_img1,(patch_size,patch_size))
        x = torch.tensor(x, dtype=torch.float).cuda()
       # x_noisy = torch.tensor(x_noisy, dtype=torch.float).cuda()
        for batch_index in range(batchs_size):
            ref_ind = np.random.randint(x.shape[1])# pick random ref patch
            x_ref = x[:, ref_ind:ref_ind + 1]
            norms = torch.norm((x_ref-x), dim=0)# norm matrix x is clean patch_image
            match_inds = torch.argsort(norms)[1:num_matches + 1] # 5 number match
            un_match_inds = torch.argsort(norms)[50:50+num_unmatches]
           # x_ref1 = x_noisy[:, ref_ind:ref_ind + 1]
            x_matched = x[:, match_inds]
            x_unmatched = x[:,un_match_inds]
           # loss_fit = torch.mean(torch.norm(hard_thresh(W @ x_ref, threshold) - hard_thresh(W @ x_matched, threshold), dim=0)) \
           #        - torch.mean(torch.norm(hard_thresh(W @ x_ref, threshold) - hard_thresh(W @ x_unmatched, threshold), dim=0))
            loss_fit = torch.mean(torch.norm(W@x_ref-W@x_matched,dim=0))-torch.mean(torch.norm(W@x_ref-W@x_unmatched,dim=0))
            loss_reg = - gamma_0 * W.slogdet()[1] + gamma_1 * torch.sum(torch.abs(W)**2)#torch.norm(W)#torch.sum((W)**2)
            loss = loss_fit + loss_reg
            step_loss += loss.item()
            step_fitloss += loss_fit.item()
           # W1 = W.detach().numpy()
           # cond.append(np.linalg.cond(W1))
            opti.zero_grad()
            loss.backward()
            opti.step()
           # P2block = hard_thresh(W@x_matched,threshold).detach().numpy()
           # sparsity.append((np.count_nonzero(P2block) / np.prod(np.shape(P2block))) * 100)
   # epoch_losses.append(step_loss)
   # epoch_fitlosses.append(step_fitloss)
    print('epoch',step)
    print('epoch_loss',step_loss)
    print('epoch_fitloss',step_fitloss)
W = W.cpu().detach().numpy()

path = r"C:\Users\james\Desktop\dataset"
np.save(os.path.join(path,'Wmatrixnormalize1.npy'),W)#WmatrixBSD002_01th80 0.3 th
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
