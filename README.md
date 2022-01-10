
## TransPath(coming soon)

The new better and stronger pre-trained models have [released](https://github.com/Xiyue-Wang/RetCCL)
#### Hardware

* 128GB of RAM
* 32*Nvidia V100 32G GPUs

### Preparations
1.Download all [TCGA](https://portal.gdc.cancer.gov/projects?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22projects.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%5D%7D) WSIs.

2.Download all [PAIP](http://wisepaip.org/paip) WSI

We crop these WSIs into patch images.we randomly select 100 images from each WSI.Finally,So, there will be about 2,700,521 unlabeled histopathological
images.If you want these images, you can contact me.

### Usage: Pre-Training Vision Transformers for histopathology images

#### 1.MoCo v3 
We also trained [MoCo v3](https://arxiv.org/abs/2104.02057) on these histopathological images.
The pre-trained  models can be downloaded as following:

[vit_small](https://drive.google.com/file/d/13d_SHy9t9JCwp_MsU2oOUZ5AvI6tsC-K/view?usp=sharing)

[vit_conv_small](https://drive.google.com/file/d/1DyavcscqiRNrZY2d2fIwFCK5FvE1WRwG/view?usp=sharing)
Undated the latest weights have been uploaded(1/10/2022)
##### Usage: Self-supervised Pre-Training
please see the [instructions](https://github.com/facebookresearch/moco-v3)

##### Usage: Get frozen features

```
python get_features_mocov3.py \
        -a vit_small
```
##### Usage: End-to-End Fine-tuning ViT
To perform end-to-end fine-tuning for ViT, use our script to convert the pre-trained ViT checkpoint to [DEiT](https://github.com/facebookresearch/deit) format:
```
python convert_to_deit.py \
  --input [your checkpoint path]/[your checkpoint file].pth.tar \
  --output [target checkpoint file].pth
```
Then run the training (in the DeiT repo) with the converted checkpoint:
```
python $DEIT_DIR/main.py \
  --resume [target checkpoint file].pth \
  --epochs 150
```

#### 2.TransPath

The pre-trained  models can be [downloaded](https://drive.google.com/file/d/1dhysqcv_Ct_A96qOF8i6COTK3jLb56vx/view?usp=sharing)

These codes are partly based on [byol](https://github.com/lucidrains/byol-pytorch) and [moco v2](https://github.com/facebookresearch/moco)
##### Usage: Self-supervised Pre-Training
```
python main_byol_transpath.py \
--lr 0.0001 \
--batch-size 256 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos
```
##### Usage: Get frozen features
```
python get_feature_transpath.py
```

##### Usage: End-to-End Fine-tuning
use our script to convert the pre-trained ViT checkpoint to Transformers format:
```
python convert_to_transpath.py 
```


## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [Pytorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models)


### Citation
Please use below to cite this paper if you find our work useful in your research.
```
@inproceedings{wang2021transpath,
  title={TransPath: Transformer-Based Self-supervised Learning for Histopathological Image Classification},
  author={Wang, Xiyue and Yang, Sen and Zhang, Jun and Wang, Minghui and Zhang, Jing and Huang, Junzhou and Yang, Wei and Han, Xiao},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={186--195},
  year={2021},
  organization={Springer}
}
``` 




