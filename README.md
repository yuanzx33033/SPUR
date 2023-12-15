# SPUR
Code and models for the paper: Self-Paced Unified Representation Learning for Hierarchical Multi-Label Classification.

<p align="center">
<<<<<<< HEAD
  <img width="950" height="270.5" src=./spur_framework.png>
=======
  <img width="950" height="230.5" src=./spur_framework.png>
>>>>>>> ee2a43605498c7033470efde118e8406110ceb73
</p>

## Requirements
To install the various Python dependencies
```
pip install -r requirements.txt
```

## Training
Training SPUR is easy! To start a particular experiment, just do
```
python main.py --dataset <dataset_name> --seed <seed_num> --device <device_num>
```
For example:
```
 python main.py --dataset cellcycle_FUN --seed 0 --device 0
```

## Acknowledgement
Thanks to others for the open-source work: <a href="https://github.com/EGiunchiglia/C-HMCNN"> C-HMCNN(h) </a> and <a href="https://github.com/weihua916/powerful-gnns"> GIN </a>

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
