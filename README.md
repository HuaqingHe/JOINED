# JOINED
This repository is an official PyTorch implementation of paper:

JOINED:  Prior guided multi-task learning for joint optic disc/cup segmentation and fovea detection. In Medical Imaging with Deep Learning MIDL (2022), 2022.

MIDL 2022
![image](https://github.com/HuaqingHe/JOINED/blob/main/Figs/fig1_pipeline.png)

## Usage
### 1. Prepare data
Please go to "./Dataset/README.md" for details.


### 2. Environment
Please prepare an environment with python >= 3.7, and the use the command ```pip install -r requirements.txt``` for the dependencies.

### 3. Train/Test

* Run the train script on GAMMA dataset. The batch size can be reduce to 8 or 4 to save memory (please also decrease the base_lr linearly), and both can reach similar performance. First coarse stage :
```bash 
cd Coarse
bash train_smp.sh
```
After Coarse stage, you need to run the `eval.sh` to get the cropped data and then run the fine stage :
```bash
cd Fine
bash train.sh
```

* Run the test script on GAMMA dataset.
First coarse stage :
```bash 
cd Coarse
bash eval.sh
```
After Coarse stage, and get the cropped data, run the fine stage :
```bash
cd Fine
bash eval.sh
```
## Citations
```bibtex
@inproceedings{he2022joined,
    title={{JOINED}: Prior Guided Multitask Learning for Joint Optic Disc/Cup Segmentation and Fovea Detection},
    author={Huaqing He and Li Lin and Zhiyuan Cai and Xiaoying Tang},
    booktitle={Medical Imaging with Deep Learning},
    year={2022},
    url={https://openreview.net/forum?id=HU6-t9oKvRW}
}
```