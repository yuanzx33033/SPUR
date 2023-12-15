# SPUR
Code and models for the paper: Self-Paced Unified Representation Learning for Hierarchical Multi-Label Classification


## Requirements
To install the various python dependencies (including tensorflow)
```
pip install -r requirements.txt
```

## Training
Training SPUR is easy!. To start a particular experiment, just do
```
python main.py --dataset <dataset_name> --seed <seed_num> --device <device_num>
```
For example:
```
 python main.py --dataset cellcycle_FUN --seed 0 --device 0
```


## Citation
If you use this code, please cite our paper
```
@inproceedings{spur2024aaai,
  title = {Self-Paced Unified Representation Learning for Hierarchical Multi-Label Classification},
  author = {Yuan, Zixuan and Liu, Hao and Zhou, Haoyi and Zhang, Denghui and Zhang, Xiao and Wang, Hao and Xiong, Hui},
  booktitle = {AAAI},
  year = 2024
}
```
