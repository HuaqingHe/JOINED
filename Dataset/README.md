# Data Preparing
1. Access to the GAMMA dataset: 
   1. Sign up in the [official GAMMA data website](https://gamma.grand-challenge.org/) and download the dataset.
   2. Run the `pre_FoveaOD.m` to get the GroundTruth of Heatmap and Distance map.
2. Fine stage must have its mask in the coarse stage before can be tested.

```bash

└── JOINED
    ├── Coarse 
    |   ├── data
    |   |   ├── train
    |   |   └── val
    |   ├── mask_DC
    |   |   ├── train
    |   |   └── val
    |   └── mask_FD_mat_0.05h
    |       ├── train
    |       └── val
    └── Fine 
        ├── DC_data
        |   ├── train
        |   └── val
        ├── mask_DC
        |   ├── train
        |   └── val
        ├── Fovea_data
        |   ├── train
        |   └── val
        └── mask_FD_mat_0.05h
            ├── train
            └── val
```