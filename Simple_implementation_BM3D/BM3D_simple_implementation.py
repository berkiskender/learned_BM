import numpy as np
import os
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

def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]
def col2im(mtx, image_size, block_size): # function change coloumn back to image
    p, q = block_size
    sx = image_size[0] - p + 1
    sy = image_size[1] - q + 1
    result = np.zeros(image_size)
    weight = np.zeros(image_size)
    col = 0

    for i in range(sx):
        for j in range(sy):
            result[i:i + q,j:j+ p] += mtx[:,col].reshape(block_size)
            weight[i:i + q, j:j + p] +=np.ones(block_size)

            col += 1
    return result / weight
def hard_thresh(x,threshold):
    # hard threshold a tensor
    y = x.copy()
    y[np.abs(y)  < threshold] = 0
    return y
#W1 = np.load((os.path.join('..','BM3D','Wmatrixurban200_80th.npy'))) # superivised matrix
W1 = np.load(os.path.join('..','BM3D','Wmatirx_1_dct_bara_1e5.npy'))
W2 = np.load(os.path.join('..','BM3D','Wmatirx_2_eye_bara_1e5.npy'))
W3 = np.load(os.path.join('..','BM3D','Wmatirx_3_eye_bara_1e5.npy'))
 
def DCT(n):
    """
    #function to produce the DCT matrix
    """
    phi = scipy.fftpack.dct(np.eye(n), norm='ortho').T
    DCT = np.kron(phi,phi)
    return DCT
#W = DCT(8)
#W = torch.tensor(W)
x_noisy  = im2col_sliding_strided(noisy,(8,8))
#x_noisy = torch.tensor(x_noisy)

patch_size = 8
#print(img)
Threshold_Hard3D = 100
PSNR_list = []
threshold_list = [90,100]
for j in range(len(threshold_list)):
    num_matches = 5
    match_inds = np.zeros([x_noisy.shape[1], num_matches]).astype(int)
    
    Hwx_noisy = hard_thresh(np.matmul(W1,np.matmul(W2,np.matmul(W3,x_noisy).clip(min=0))),threshold_list[j])
    for ref_ind in range(x_noisy.shape[1]):
        x_ref1 = x_noisy[:, ref_ind:ref_ind+1]
        loss_patch_sorted = np.argsort(
        np.linalg.norm(hard_thresh(np.matmul(W1,np.matmul(W2,np.matmul(W3,x_ref1).clip(min=0))),threshold_list[j]) - Hwx_noisy, axis=0))
        match_inds[ref_ind, :] = loss_patch_sorted[1:num_matches + 1]
#torch.save(os.path.join(r'C:\Users\james\Desktop\dataset\numpy_matchind_bara.npy'),match_inds)



    reconstruct_image = np.zeros(x_noisy.shape) 
    for i in range(match_inds.shape[0]):
        ref_ind = i
    #print(match_inds[ref_ind, :][0])
        ref_patch = x_noisy[:,ref_ind:ref_ind+1].reshape(patch_size,patch_size,1)
        simliar_patch = x_noisy[:,match_inds[ref_ind,:]].reshape(patch_size,patch_size,num_matches)
        thereDtensor = np.concatenate((ref_patch, simliar_patch), axis =2) # build up the tensor that composed of the simliar patch and ref patch
        #thereDtensor = np.vectorize(thereDtensor)
        thereDtensor = scipy.fftpack.dct(thereDtensor,axis =0,norm ='ortho')
        thereDtensor = scipy.fftpack.dct(thereDtensor,axis =1,norm ='ortho')
        thereDtensor = scipy.fftpack.dct(thereDtensor,axis =2,norm ='ortho') # dct transform the tensor 
        thereDtensor[np.abs(thereDtensor[:]) < 120] = 0. # hard threshold the transform tensor
        thereDtensor = scipy.fftpack.idct(thereDtensor,axis =2,norm='ortho')
        thereDtensor = scipy.fftpack.idct(thereDtensor,axis =1,norm='ortho')
        thereDtensor = scipy.fftpack.idct(thereDtensor,axis =0,norm ='ortho') # inverse transform the tensor
    #print(thereDtensor.shape)
       # print(thereDtensor[:,:,0])
        output = thereDtensor[:,:,0].reshape(64)
       # print(output)
        reconstruct_image[:,i] = output
        output2 = thereDtensor[:,:,1:6].reshape(64,5)
        reconstruct_image[:,match_inds[i,:]] = output2
    reconstruct_image = col2im(reconstruct_image,(512,512),(8,8))
    print(compute_psnr(img,reconstruct_image))
    PSNR_list.append(compute_psnr(img,reconstruct_image))
    
np.save(os.path.join('..','BM3D','PSNR_Transform__matrix_BSD_img001.npy'),PSNR_list)

plt.plot(threshold_list,PSNR_list)
plt.show()
