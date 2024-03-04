<div align="center">
  <h1>SRML: Structure-Relation Mutual Learning Network for Few-shot Image Classification <br> </h1>
</div>


## :heavy_check_mark: Requirements
* Ubuntu 16.04
* Python 3.7
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.1](https://pytorch.org)


## :gear: Conda environmnet installation
```bash
conda env create --name srml --file environment.yml
conda activate srml
```

## :deciduous_tree: Authors' checkpoints

The file structure should be as follows:


    
    SRML/
    ├── datasets/
    ├── common/
    ├── models/
    ├── checkpoints/
    │   ├── cub/
    train.py
    test.py
    README.md
    environment.yml