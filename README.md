# LABMAT

Implementation of *LABMAT: Learned Feature-Domain Block Matching For Image Restoration* ([IEEE ICIP](https://ieeexplore.ieee.org/abstract/document/9506629))

**Authors**: Shijun Liang $^\*$ ; Berk Iskender $^\*$ ; Bihan Wen; Saiprasad Ravishankar

**Abstract**: Grouping of similar patches, called block matching, has been widely used in image restoration applications. 
Popular block matching algorithms exploit image non-local similarities in spatial or a fixed transform domain, e.g., wavelets and DCT. 
However, applying these methods on corrupted patches usually leads to degraded matching accuracy, thus limiting the image restoration performance. 
In this work, we develop a novel methodology for performing block matching in a supervised way by learning multi-layer sparsifying transforms. 
The proposed learned transform-domain block matching method for image restoration, dubbed LABMAT, 
is shown to have better accuracy in terms of clustering similar blocks in the presence of noise, 
and it also achieves an improved denoising performance when it is incorporated into popular non-local denoising schemes.

![alt text](https://github.com/berkiskender/learned_BM/blob/master/labmat.png)

$^*$ WLBM: [WNNM](https://openaccess.thecvf.com/content_cvpr_2014/papers/Gu_Weighted_Nuclear_Norm_2014_CVPR_paper.pdf) with LABMAT

## Citation
If you find this work useful for your research, please cite:

*S. Liang, B. Iskender, B. Wen and S. Ravishankar, "Labmat: Learned Feature-Domain Block Matching For Image Restoration," 2021 IEEE International Conference on Image Processing (ICIP), Anchorage, AK, USA, 2021, pp. 1689-1693, doi: 10.1109/ICIP42928.2021.9506629.*

```
@inproceedings{liang2021labmat,
  title={Labmat: Learned feature-domain block matching for image restoration},
  author={Liang, Shijun and Iskender, Berk and Wen, Bihan and Ravishankar, Saiprasad},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={1689--1693},
  year={2021},
  organization={IEEE}
}
```

## Contact
In case of any questions, feel free to contact via email: Berk Iskender, berki2@illinois.edu; Shijun Liang, liangs16@msu.edu
