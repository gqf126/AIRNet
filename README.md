# Real-Time Deep Image Retouching Based on Learnt Semantics Dependent Global Transforms


## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard:
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`


## Datasets
[MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/)

We also recommand to use the preprocessed datasets  [MIT-Adobe FiveK preprocessed dataset](https://drive.google.com/drive/folders/1qrGLFzW7RBlBO1FqgrLPrq9p2_p11ZFs?usp=sharing) (kindly shared by Jingwen He, author of CSRNet)


## How to Test
1. Modify the configuration file [`options/test/test_Meta_Y_UV_Para.json`](codes/options/test/test_Enhance.yml). 
2. Run command:
```
python test.py -opt options/test/test_Meta_Y_UV_Para.json
```


## Acknowledgement

- This code is based on [mmsr](https://github.com/open-mmlab/mmsr). 


### BibTex
    @article{gao2021real,
      title={Real-time deep image retouching based on learnt semantics dependent global transforms},
      author={Gao, Qifan and Wu, Xiaolin},
      journal={IEEE Transactions on Image Processing},
      volume={30},
      pages={7378--7390},
      year={2021},
      publisher={IEEE}}
