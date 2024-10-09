## [AFANet: A Multibackbone Compatible Feature Fusion Framework for Effective Remote Sensing Object Detection](https://ieeexplore.ieee.org/document/10681114)

### 1. Installation

For installation instructions, check the [mmdetection](https://github.com/open-mmlab/mmdetection)/[mmyolo](https://github.com/open-mmlab/mmyolo) github site. The environment required for this letter will set up according to the official MM steps.



### 2. Overview architecture of the proposed AFANet

​	With the multiple low-to-high levels of preliminary feature maps {C2, C3, C4, C5} generated from an input remote sensing image via a backbone network, as depicted in Fig. 1, AFANet leverages the MASA module on the high-level feature map C5 to extract richer semantic information Cout. This process benefits the capture of global contextual semantics for precise pattern recognition, particularly for small objects. Next, the SGCFF block combines the semantic feature Cout with the other feature scales from the backbone, producing refined feature representations {G2, G3, G4, G5} incorporating both local and global information. The inclusion of self-attention transformer and deformable convolution applied across all feature levels within the SGCFF block enhances the robustness of AFANet to detect divers-scale and irregular objects, regardless of the backbone used. To facilitate the adaptive selection of more regions for diverse scales of remote objects, the attention head (AH) employs max pooling with a channel-spatial attention mechanism (CBAM)  to downsample G5, creating the highest-level output feature map G6. The dual-attention mechanism in AH can alleviate information loss and spatial distortion during feature scale expansion, ultimately improving the network’s ability to capture intricate details and precise location cues. Note that the backbone architecture is not strictly constrained in this work.

![image-20241009185058757](https://raw.githubusercontent.com/mf991/typora-images/main/image-20241009185058757.png)




### 3. Some experimental results of this letter



<img src="https://raw.githubusercontent.com/mf991/typora-images/main/image-20241009171517299.png" alt="image-20241009171517299" style="zoom: 67%;" />





<img src="https://raw.githubusercontent.com/mf991/typora-images/main/image-20241009171537669.png" alt="image-20241009171537669" style="zoom:67%;" />





<img src="https://raw.githubusercontent.com/mf991/typora-images/main/image-20241009171557194.png" alt="image-20241009171557194" style="zoom:67%;" />





### 4. Please cite our paper if it is useful for you. Thank you!

Q. Yi, M. Zheng, M. Shi, J. Weng and A. Luo, "AFANet: A Multibackbone Compatible Feature Fusion Framework for Effective Remote Sensing Object Detection," in *IEEE Geoscience and Remote Sensing Letters*, vol. 21, pp. 1-5, 2024, Art no. 6015805, doi: 10.1109/LGRS.2024.3462089.   [link](https://ieeexplore.ieee.org/document/10681114)


```
@ARTICLE{10681114,
  author={Yi, Qingming and Zheng, Mingfeng and Shi, Min and Weng, Jian and Luo, Aiwen},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={AFANet: A Multibackbone Compatible Feature Fusion Framework for Effective Remote Sensing Object Detection}, 
  year={2024},
  volume={21},
  pages={1-5}}
```
