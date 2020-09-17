import numpy as np
import scipy.spatial
import os
#import mat73
import cv2
import torch
import math
import scipy
import scipy.fftpack
import matplotlib.pyplot as plt

#img = np.random.uniform(low=0,high=256,size=(64, 64))
#np.save(os.path.join(r'C:\Users\james\Desktop\dataset\patchimage.npy'),img)
#img = cv2.imread(os.path.join('..','residue_data','Train400','test_001.png'))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float)
img = np.load(os.path.join('..','BM3D','barbara_512x512.npy')).astype(float)
noisy = np.load(os.path.join('..','BM3D','barbara_512x512_noisy.npy')).astype(float)
#img = np.load(os.path.join('..','residue_data','image_SRF_2_reshape','001.npy')).astype(float)
#noisy = np.load(os.path.join('..','residue_data','image_SRF_2_noise_sigma_50','001.npy')).astype(float)
def compute_psnr(img1, img2):
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * math.log10(1. / mse)

def im2col1(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]
def make_inds(im_sz, sz):
    dm, dn = np.meshgrid(range(sz[0]), range(sz[1]))
    m0, n0 = np.meshgrid(range(im_sz[0]-sz[0]+1), range(im_sz[1]-sz[1]+1))

    M = dm.flatten()[:, np.newaxis] + m0.flatten()[np.newaxis, :]
    N = dn.flatten()[:, np.newaxis] + n0.flatten()[np.newaxis, :]

    return M, N


def im2col(im, sz):
    """
    im = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
    X = im2col(im, [2, 2])
    np.testing.assert_array_equal(X[:, 1], [4, 7, 5, 8])

    """
    im_sz = im.shape
    M, N = make_inds(im_sz, sz)
    X = im[M, N]

    return X
def col2im(col, im_sz, sz):
    M, N = make_inds(im_sz, sz)

    i = np.ravel_multi_index((M, N), im_sz)
    counts = np.bincount(i.flat)
    vals = np.bincount(i.flat, weights=col.flat)

    return np.reshape(vals/counts, im_sz)
#def col2im(mtx, image_size, block_size): # function change coloumn back to image
#    p, q = block_size
#    sx = image_size[0] - p + 1
#    sy = image_size[1] - q + 1
#    result = np.zeros(image_size)
#    weight = np.zeros(image_size)
#    col = 0

#    for i in range(sx):
#        for j in range(sy):
#            result[i:i + q,j:j+ p] += mtx[:,col].reshape(block_size)
#            weight[i:i + q, j:j + p] +=np.ones(block_size)

#            col += 1
#    return result / weight
def hard_thresh(x,threshold):
    # hard threshold a tensor
    y = x.clone()
    y[y.abs()  < threshold] = 0
    return y
torch.cuda.set_device(0)
#W1 = np.load((os.path.join('..','BM3D','Wmatrixurban200_80th.npy'))) # superivised matrix
W2 = np.load(os.path.join('..','BM3D','Wmatrix2_bug_fix_90_th_1e5.npy'))
W1 = np.load(os.path.join('..','BM3D','Wmatrix1_bug_fix_90_th_1e5.npy'))
W3 = np.load(os.path.join('..','BM3D','Wmatrix3_bug_fix_90_th_1e5.npy'))
#W2 = np.load(os.path.join('..','BM3D','W2Matrix5_match_10_unmatch_3e-5_lr_Bar_512.npy'))
#W1 = np.load(os.path.join('..','BM3D','Matrix5_match_10_unmatch_3e-5_lr_Bar_512.npy'))

W1 =torch.tensor(W1).cuda().float()
W2 = torch.tensor(W2).cuda().float()
W3 = torch.tensor(W3).cuda().float()
def DCT(n):
    """
    #function to produce the DCT matrix
    """
    phi = scipy.fftpack.dct(np.eye(n), norm='ortho').T
    DCT = np.kron(phi,phi)
    return DCT
W_transform = np.load(os.path.join('..','BM3D','Wmatrixwindowsize1e-3lr.npy'))
W_transform = torch.tensor(W_transform).cuda().float()
W_dct = DCT(8)
W_dct = torch.tensor(W_dct).cuda().float()
x_noisy  = im2col(noisy,[8,8])#.astype(np.float)
x_noisy = torch.tensor(x_noisy).cuda().float()
x = im2col(img,[8,8])
x = torch.tensor(x).cuda().float()
x_noisy_numpy = im2col(noisy,[8,8])
patch_size = 8
#print(img)
Threshold_Hard3D = 100
PSNR_list = []
threshold_list = [70,80,90,100]

for j in range(len(threshold_list)):
    print('processing')
    num_matches = 5
    match_inds = np.zeros([x_noisy.shape[1], num_matches]).astype(np.int)
    match_inds = torch.tensor(match_inds).cuda().long()
   # Hwx_noisy = hard_thresh(x_noisy,threshold_list[j])
   # Hwx_noisy = hard_thresh(W1@(W2@(W3@x_noisy)),threshold_list[j])
    Hwx_noisy = hard_thresh((W_transform@x_noisy),threshold_list[j])
    for ref_ind in range(x_noisy_numpy.shape[1]):
        x_ref1 = x_noisy[:, ref_ind:ref_ind+1]
       # loss_patch_sorted = torch.argsort(torch.norm(hard_thresh(W1@(W2@(W3@x_ref1)),threshold_list[j]) - Hwx_noisy, dim=0))
        loss_patch_sorted = torch.argsort(torch.norm(hard_thresh((W_transform@x_ref1),threshold_list[j]) - Hwx_noisy,dim=0))
       # loss_aptch_sorted = torch.argosrt(torch.norm(x_ref1 -x,dim=0))
        match_inds[ref_ind, :] = loss_patch_sorted[1:num_matches + 1]
#torch.save(os.path.join(r'C:\Users\james\Desktop\dataset\numpy_matchind_bara.npy'),match_inds)

    match_inds_numpy = match_inds.cpu().detach().numpy()
    reconstruct_image = np.zeros(x_noisy_numpy.shape)
    reconstruct_image_count = np.zeros(x_noisy_numpy.shape)
    print('block matching done')
    for i in range(match_inds.shape[0]):
        ref_ind = i
    #print(match_inds[ref_ind, :][0])
        ref_patch = x_noisy_numpy[:,ref_ind:ref_ind+1].reshape(patch_size,patch_size,1)
        simliar_patch = x_noisy_numpy[:,match_inds_numpy[ref_ind,:]].reshape(patch_size,patch_size,num_matches)
        thereDtensor = np.concatenate((ref_patch, simliar_patch), axis =2) # build up the tensor that composed of the simliar patch and ref patch
        #thereDtensor = np.vectorize(thereDtensor)
        thereDtensor = scipy.fftpack.dct(thereDtensor,axis =0,norm ='ortho')
        thereDtensor = scipy.fftpack.dct(thereDtensor,axis =1,norm ='ortho')
        thereDtensor = scipy.fftpack.dct(thereDtensor,axis =2,norm ='ortho') # dct transform the tensor 
        thereDtensor[np.abs(thereDtensor[:]) < Threshold_Hard3D] = 0. # hard threshold the transform tensor
        thereDtensor = scipy.fftpack.idct(thereDtensor,axis =2,norm ='ortho')
        thereDtensor = scipy.fftpack.idct(thereDtensor,axis =1,norm ='ortho')
        thereDtensor = scipy.fftpack.idct(thereDtensor,axis =0,norm ='ortho') # inverse transform the tensor
    #print(thereDtensor.shape)
       # print(thereDtensor[:,:,0])
        output = thereDtensor[:,:,0].reshape(64)
       # print(output)
        reconstruct_image[:,i] += output
        output2 = thereDtensor[:,:,1:6].reshape(64,5)
        reconstruct_image_count[:,i] += 1
        reconstruct_image[:,match_inds_numpy[i,:]] += output2
        reconstruct_image_count[:,match_inds_numpy[i,:]] += 1
    reconstruct_image = reconstruct_image/reconstruct_image_count
    reconstruct_image_true = col2im(reconstruct_image,noisy.shape,[8,8])
    print(compute_psnr(img,reconstruct_image_true))
    PSNR_list.append(compute_psnr(img,reconstruct_image_true))
    
np.save(os.path.join('..','BM3D','PSNR_Transform__matrix_BSD_img001.npy'),PSNR_list)

plt.plot(threshold_list,PSNR_list)
plt.show()
