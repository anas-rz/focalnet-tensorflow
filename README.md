# A TensorFlow translation of FocalNet (WIP)

Released by Microsoft in 2022, FocalNet or Focal Modulation Network is an attention-free architecture achieving superior performance than SoTA self-attention (SA) methods across various vision benchmarks.  [GitHub](https://github.com/microsoft/FocalNet/) [Paper](https://arxiv.org/abs/2203.11926).

# Installation

```
pip install focalnet-tensorflow
```

# Usage

```
from focalnet_tensorflow import *
model = focalnet_huge_fl3()
```
See usage [here](./test_notebook.ipynb)

# Available Functions:


```
focalnet_tiny_srf

focalnet_small_srf

focalnet_base_srf

focalnet_tiny_lrf

focalnet_small_lrf

focalnet_base_lrf

focalnet_tiny_iso_16

focalnet_small_iso_16

focalnet_base_iso_16

focalnet_large_fl3

focalnet_large_fl4

focalnet_xlarge_fl3

focalnet_xlarge_fl4

focalnet_huge_fl3

focalnet_huge_fl4
```
