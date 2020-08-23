# PHYSICAL-MODEL-GUIDED-DEEP-IMAGE-DERAINING

Honghe Zhu\*, [Cong Wang](https://supercong94.wixsite.com/supercong94)\*, Yajie Zhang, [Zhixun Su](http://faculty.dlut.edu.cn/ZhixunSu/zh_CN/index/759047/list/index.htm) †, Guohui Zhao 

<\* Both authors contributed equally to this research. † Corresponding author.>

This work has been accepted by ICME 2020.

## Abstract
Single image deraining is an urgent task because the degraded rainy image makes many computer vision systems fail to work, such as video surveillance and autonomous driving. So, deraining becomes important and an effective deraining algorithm is needed. In this paper, we propose a novel network based on physical model guided learning for single image deraining, which consists of three sub-networks: rain streaks network, rain-free network, and guide-learning network. The concatenation of rain streaks and rain-free image that are estimated by rain streaks network, rain-free network, respectively, is input to the guide-learning network to guide further learning and the direct sum of the two estimated images is constrained with the input rainy image based on the physical model of rainy image. Moreover, we further develop the Multi-Scale Residual Block (MSRB) to better utilize multi-scale information and it is proved to boost the deraining performance. Quantitative and qualitative experimental results demonstrate that the proposed method outperforms the state-of-the-art deraining methods.

## Requirements
- CUDA 9.0
- Python 3.6 (or later)
- Pytorch 1.1.0
- Torchvision 0.3.0
- OpenCV

## Dataset
Please download the following datasets:

* Rain100L [[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Rain100H [[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Rain1200 [[dataset](https://github.com/hezhangsprinter/DID-MDN)]

## Setup
Please download this project through 'git' command.
```
$ git clone https://github.com/Ohraincu/PHYSICAL-MODEL-GUIDED-DEEP-IMAGE-DERAINING.git
$ cd config
```

Thanks to [the code by Li et al.](https://xialipku.github.io/RESCAN/), our code is also adapted based on this.

## Training
After you download the above datasets, you can perform the following operations to train:
```
$ python train.py
```  
You can pause or start the training at any time because we can save the pre-trained models in due course.

## Testing
### Pre-trained Models
Wait update!

### Quantitative and Qualitative Results
After running eval.py, you can get the corresponding numerical results (PSNR and SSIM):
```
$ python eval.py
``` 
If the visual results on datasets need to be observed, the show.py can be run:
```
$ python show.py
``` 

## Citation
```
@inproceedings{derain_zhu_icme,
  author    = {Honghe Zhu and
               Cong Wang and
               Yajie Zhang and
               Zhixun Su and
               Guohui Zhao},
  title     = {Physical Model Guided Deep Image Deraining},
  booktitle = {{IEEE} International Conference on Multimedia and Expo, {ICME} 2020,
               London, UK, July 6-10, 2020},
  pages     = {1--6},
  year      = {2020},
}
```

## Contact

Email: supercong94@gmail.com
