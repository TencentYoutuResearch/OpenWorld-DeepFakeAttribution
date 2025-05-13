# Open-World DeepFake Attribution

> This repository is official implementation for Contrastive Pseudo Learning for Open-World DeepFake Attribution, ICCV 2023 and Rethinking Open-World DeepFake Attribution with Multi-perspective Sensory Learning, IJCV 2024

![python](https://img.shields.io/badge/python-3.9-blue?logo=python)
![pytorch](https://img.shields.io/badge/pytorch-1.13.1-blue?logo=pytorch)
![lightning](https://img.shields.io/badge/lightning-Enabled-enabled?logo=lightning)

## Overview

The challenge in sourcing attribution for forgery faces has gained widespread attention due to the rapid development of generative techniques. While many recent works have taken essential steps on GAN-generated faces, more threatening attacks related to identity swapping or expression transferring are still overlooked. And the forgery traces hidden in unknown attacks from the open-world unlabeled faces still remain under-explored.
To push the related frontier research, we introduce a new benchmark called Open-World DeepFake Attribution (OW-DFA), which aims to evaluate attribution performance against various types of fake faces under open-world scenarios.
Meanwhile, we propose a novel framework named Contrastive Pseudo Learning (CPL) for the OW-DFA task through 1) introducing a Global-Local Voting module to guide the feature alignment of forged faces with different manipulated regions, 2) designing a Confidence-based Soft Pseudo-label strategy to mitigate the pseudo-noise caused by similar methods in unlabeled set. In addition, we extend the CPL framework with a multi-stage paradigm that leverages pre-train technique and iterative learning to further enhance traceability performance.
Extensive experiments verify the superiority of our proposed method on the OW-DFA and also demonstrate the interpretability of deepfake attribution task and its impact on improving the security of deepfake detection area.


## Dataset

- Prepare Deepfake Detection datasets

  |     Dataset     |                                 Paper                                 |                                                 Link                                                  |
  | :-------------: | :-------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
  | FaceForensics++ |     FaceForensics++: Learning to Detect Manipulated Facial Images     |      [Paper](https://arxiv.org/abs/1901.08971) [Code](https://github.com/ondyari/FaceForensics)       |
  |    Celeb-DF     |  Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics   | [Paper](https://arxiv.org/abs/1909.12962) [Code](https://github.com/yuezunli/celeb-deepfakeforensics) |
  |   ForgeryNet    | ForgeryNet: A Versatile Benchmark for Comprehensive Forgery Analysis  | [Paper](https://arxiv.org/abs/2103.05630) [Home](https://yinanhe.github.io/projects/forgerynet.html)  |
  |      DFFD       |             On the Detection of Digital Face Manipulation             |     [Paper](https://arxiv.org/abs/1910.01717) [Home](https://cvlab.cse.msu.edu/project-ffd.html)      |
  |   ForgeryNIR    | ForgeryNIR: Deep Face Forgery and Detection in Near-Infrared Scenario |  [Paper](https://ieeexplore.ieee.org/document/9693897) [Code](https://github.com/AEP-WYK/forgerynir)  |
  |      DF^3       | GLFF: Global and Local Feature Fusion for AI-synthesized Image Detection |  [Paper](https://arxiv.org/abs/2211.08615) [Code](https://github.com/littlejuyan/GLFF)  |

- Download dataset and unzip data under the directory of `/Datasets/deepfakes_detection_datasets/`
- Process dataset with script `scripts/preprocess/create_academic_meta.ipynb`, and you will get the following structure:

  ```bash
  data/release
  ├── AttributeManipulation
  │   ├── FaceAPP
  │   │   └── DFFD
  │   ├── MaskGAN
  │   │   └── ForgeryNet
  │   ├── SC-FEGAN
  │   │   └── ForgeryNet
  │   ├── StarGAN
  │   │   └── DFFD
  │   └── StarGAN2
  │       └── ForgeryNet
  ├── EntireFaceSyncthesis
  │   ├── CycleGAN
  │   │   └── ForgeryNIR
  │   ├── PGGAN
  │   │   └── DFFD
  │   ├── StyleGAN
  │   │   └── DFFD
  │   └── StyleGAN2
  │       ├── ForgeryNet
  │       └── ForgeryNIR
  ├── ExpressionTransfer
  │   ├── ATVG-Net
  │   │   └── ForgeryNet
  │   ├── Face2Face
  │   │   └── faceforensics
  │   ├── FOMM
  │   │   └── ForgeryNet
  │   ├── NeuralTextures
  │   │   └── faceforensics
  │   └── Talking-Head-Video
  │       └── ForgeryNet
  ├── IdentitySwap
  │   ├── DeepFaceLab
  │   │   └── ForgeryNet
  │   ├── Deepfakes
  │   │   └── faceforensics
  │   ├── FaceShifter
  │   │   └── ForgeryNet
  │   ├── FaceSwap
  │   │   └── faceforensics
  │   └── FSGAN
  │       └── ForgeryNet
  ├── RealFace
  │   └── Real
  │       ├── CelebDF
  │       └── faceforensics
  ├── meta_data
  │   ├── Protocol1_openset_fake_large_merge_meta.csv
  │   ├── Protocol1_openset_fake_val_merge_meta.csv
  │   ├── Protocol2_openset_real_fake_large_merge_meta.csv
  │   └── Protocol2_openset_real_fake_val_merge_meta.csv
  └── shape_predictor_68_face_landmarks.dat
  ```

## Method

<img src="images/CPL.png" alt="CPL" width=1000 />

## Quick Start
**Step1.** Create a conda environment and activate it.
```bash
conda create --name owdfa python=3.9 -y
conda activate owdfa
```

**Step2.** Install the required python libraries.
```bash
cd OW-DFA
pip3 install -r requirements.txt
wandb offline
```

**Step3.** Train MPSL model on OW-DFA dataset.

```bash
python3 -u -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 \
  train.py -c configs/train.yaml
```

## Citation
If you find this project useful in your research, please consider cite:
```bibtex
@inproceedings{sun2023contrastive,
  title={Contrastive pseudo learning for open-world deepfake attribution},
  author={Sun, Zhimin and Chen, Shen and Yao, Taiping and Yin, Bangjie and Yi, Ran and Ding, Shouhong and Ma, Lizhuang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20882--20892},
  year={2023}
}

@article{sun2025rethinking,
  title={Rethinking open-world deepfake attribution with multi-perspective sensory learning},
  author={Sun, Zhimin and Chen, Shen and Yao, Taiping and Yi, Ran and Ding, Shouhong and Ma, Lizhuang},
  journal={International Journal of Computer Vision},
  volume={133},
  number={2},
  pages={628--651},
  year={2025},
  publisher={Springer}
}
```
