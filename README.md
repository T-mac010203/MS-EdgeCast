# MS-EdgeCast  
**A Dual-Stage Framework With a Multiscale Convolutional Recurrent Network and Edge-Guided Diffusion for Convective Storm Nowcasting**

---
## Citation
If you use this code or find it helpful in your research, please cite:

W. Zhang, L. Wang, M. Luo, R. Pang, X. Song and X. Zhang, "MS-EdgeCast: A Dual-Stage Framework With a Multiscale Convolutional Recurrent Network and Edge-Guided Diffusion for Convective Storm Nowcasting," in IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1-13, 2026.

doi: 10.1109/TGRS.2026.3655487

```bibtex
@article{zhang2026msedgecast,
  author={Zhang, Wei and Wang, Lintao and Luo, Muqi and Pang, Renbo and Song, Xiaojiang and Zhang, Xiangguang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MS-EdgeCast: A Dual-Stage Framework With a Multiscale Convolutional Recurrent Network and Edge-Guided Diffusion for Convective Storm Nowcasting}, 
  year={2026},
  volume={64},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2026.3655487}}
}
```
## Inference
Repository Structure
```
.
├── mscrn/                  # MS-CRN predictor (Stage I)
├── edgeguided/             # Edge-guided diffusion model (Stage II)
├── checkpoint/             # trained model checkpoints
├── radar.py                # MRMS radar dataset loader
├── config.py               # Configuration file
├── infer.py                # Inference script
├── draw_radar.py
├── metrics.py
└── README.md

```

1.Modify key runtime parameters (such as device, checkpoint paths, etc.) in the config.py file.

2.Prepare the checkpoints by downloading and extracting the pretrained checkpoints into the directory specified by cfg.ckpt_dir. 
Download Link (https://drive.google.com/file/d/1RL9UrLoDKrBZ-jrJ6K3VagDWN0BV-j_J/view?usp=sharing)

3.Run Inference
```
python infer.py
```


## Dataset: MRMS Radar Data

The inference pipeline is designed for the MRMS (Multi-Radar Multi-Sensor) dataset.(https://mtarchive.geol.iastate.edu/)


The released inference script supports the following two geographic regions:

Region 1
Longitude: −79.3 ~ −69.8
Latitude: 36.25 ~ 45.75

Region 2
Longitude: −97.3 ~ −87.8
Latitude: 27.25 ~ 36.75
