# InpaintReward

This project implements a reward model designed for image inpainting tasks, aiming to evaluate the quality of inpainted results based on learned representations.

---

## Dataset Preparation

1. Download the training dataset from [this Google Drive link](https://drive.google.com/drive/folders/1A0hdSiYmGv7OMWuJGefbAMm-UJmoKsnU?usp=sharing).
2. Place the dataset contents in the `./Data/` directory.

---

## 🚀 Usage

### 1. Clone the repository

```bash
cd ImageReward
In the YAML file located at `configs/imagereward_train_configs.yaml`, replace: ../Data/ with your own path
cd train
Run bash ./scripts/train_one_node.sh