# [Information Fusion 2025] Code for Cross-Modal Guided and Refinement-Enhanced Retinex Network for Robust Low-Light Image Enhancement [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253525004531)

## Abstract

The Retinex theory has long been a cornerstone in the field of low-light image enhancement, garnering significant attention. However, traditional Retinex-based methods often suffer from insufficient robustness to noise interference, necessitating the introduction of additional regularization terms or handcrafted priors to improve performance. These handcrafted priors and regularization-based approaches, however, lack adaptability and struggle to handle the complexity and variability of low-light environments effectively. To address these limitations, this paper proposes a Cross-Modal Guided and Refinement-Enhanced Retinex Network (CMRetinexNet) that leverages the adaptive guidance potential of auxiliary modalities and incorporates refinement modules to enhance Retinex decomposition and synthesis. Specifically: (a) Considering the characteristics of the reflectance component, we introduce auxiliary modal information to adaptively improve the accuracy of reflectance estimation. (b) For the illumination component, we design a reconstruction module that combines local and frequency-domain information, to iteratively enhance both regional and global illumination levels. (c) To address the inherent uncertainty in the element-wise multiplication of reflectance and illumination components during Retinex synthesis, we propose a synthesis and refinement module that effectively fuses illumination and reflectance components by leveraging cross-channel and spatial contextual information. Extensive experiments on multiple public datasets demonstrate that the proposed model achieves significant improvements in both qualitative and quantitative metrics compared to state-of-the-art methods, validating its effectiveness and superiority in low-light image enhancement.


### Train
```
python train.py -opt ./options/train/huawei.yml
```

### Test
```
python test.py -opt ./options/test/huawei.yml
```
