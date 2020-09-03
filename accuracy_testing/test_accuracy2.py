import scipy
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt

def convert_range(image, image_min, image_max, target_min, target_max):
    """
    :return reshape the image to range between 1 and -1:
    """
    old_range = image_max - image_min
    new_range = target_max - target_min
    target_image = np.divide(((image-image_min) * new_range),old_range) + target_min
    return target_image

# def im2col(mtx, block_size): # functinon change image to column
#     mtx_shape = mtx.shape
#     sx = mtx_shape[0] - block_size[0] + 1
#     sy = mtx_shape[1] - block_size[1] + 1
#     result = np.empty((block_size[0] * block_size[1], sx * sy))
#     for i in range(sy):
#         for j in range(sx):
#             result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].flatten()
#     return result

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
    y = x.copy()
    y[np.abs(y)< threshold] = 0
    return y

def test_accuracy(img,noisy,ref_patchs,ref_noises,W,num_matches,threshold):
    Hwx_noisy = hard_thresh(np.matmul(W,noisy),threshold)
    scores = np.zeros(ref_patchs.shape[1])
    for ref_ind in range(ref_patchs.shape[1]):
        Ref_patch = ref_patchs[:,[ref_ind]]
        norms_matrix = np.linalg.norm(Ref_patch - img, axis=0)  # norm matrix
        match_inds = np.argsort(norms_matrix)[1:num_matches + 1]
        Ref_noisy = ref_noises[:,[ref_ind]]
        noisy_norm_matrix = np.linalg.norm(hard_thresh(np.matmul(W,Ref_noisy),threshold)-Hwx_noisy,axis=0)
        noisy_match_inds = np.argsort(noisy_norm_matrix)[1:num_matches+1]
        scores[ref_ind] = len(set(match_inds).intersection(set(noisy_match_inds)))
    return scores

def DCT(n):
    """
    #function to produce the DCT matrix
    """
    phi = scipy.fftpack.dct(np.eye(n), norm='ortho').T
    DCT = np.kron(phi,phi)
    return DCT

img = np.load(r'/barbara_512x512.npy')
img = img.astype(float)
noisy_img = np.load(r'/barbara_512x512_noisy.npy')
noisy_img = noisy_img.astype(float)

# Normalize the img
# x_im = convert_range(img, np.amin(img), np.amax(img), -1, 1)
# x_im_noisy = convert_range(noisy_img, np.amin(noisy_img), np.amax(noisy_img), -1, 1)

# DO NOT normalize the img.
x_im = img
x_im_noisy = noisy_img

# Patchify the images
patch_size = 8
x = im2col(x_im, (patch_size, patch_size))
x_noisy = im2col(x_im_noisy, (patch_size, patch_size))

# Load the learned transform matrix
W = np.load(
    r'/results/Wmatrix_supervised_w_matched_5_unmatched_5_DCT_init_threshold_100.00_gamma0_500.00_refpatches_255025_subset_10000_BARBARA_512x512_LATEST.npy')
W1 = DCT(8) # Load the DCT matrix
number_patch = 1000 # Number of patches to check accuracy on
number_match = 5 # Number of matches for each patch
X_ref_patchs = np.zeros((64,number_patch))
Xnoisy_ref_patchs = np.zeros((64,number_patch))

index_matrix = np.arange(0,x.shape[1],1)
# subset_index_matrix = np.random.choice(index_matrix, size = number_patch) # Select a random subset among all patches
# np.save('/home/berk/Desktop/Internship/LANL_MSU/MSU/learned_BM/results/accuracy/subset_idx_for_acc_plots.npy', subset_index_matrix)
subset_index_matrix = np.load(r'/results/accuracy/subset_idx_for_acc_plots.npy')

# Assign matched patches for all reference patches in the "subset" of interest
for i in range(number_patch):
    ref_ind = subset_index_matrix[i]
    x_ref = x[:, ref_ind:ref_ind + 1]
    x_ref1 = x_noisy[:, ref_ind:ref_ind + 1]
    X_ref_patchs[:,[i]] = x_ref
    Xnoisy_ref_patchs[:,[i]] = x_ref1

score_array = []
score2_array = []
thresholds = [60,70,80,90,100,110,120,130,140] # Thresholds to check accuracy on

for j in range(len(thresholds)):
    print(j)
    score = test_accuracy(x,x_noisy,X_ref_patchs,Xnoisy_ref_patchs,W,number_match,thresholds[j])
    score2 = test_accuracy(x,x_noisy,X_ref_patchs,Xnoisy_ref_patchs,W1,number_match,thresholds[j])
    print('Tl accuracy',np.sum(score)/(number_patch*number_match))
    print('DCT accuracy',np.sum(score2)/(number_patch*number_match))
    score_array.append(np.sum(score))
    score2_array.append(np.sum(score2))

fig = plt.figure()
plt.plot(thresholds, np.array(score_array)/(number_patch*number_match), label = 'Transform')
plt.plot(thresholds, np.array(score2_array)/(number_patch*number_match), label = 'DCT')
plt.xlabel('threshold')
plt.ylabel('accuracy')
plt.legend()
plt.show()