# InpaintReward

This project implements a reward model designed for image inpainting tasks, aiming to evaluate the quality of inpainted results based on learned representations.

---

## Dataset Preparation

1. Download the training dataset from this [Google Drive link](https://drive.google.com/drive/folders/1A0hdSiYmGv7OMWuJGefbAMm-UJmoKsnU?usp=sharing).
2. Place the dataset contents in the `./Data/` directory.

---

## 🚀 Usage

```bash
cd ImageReward
cd train
In the YAML file located at `src/config/config.yaml`, replace: ../Data/ with your own path
Run bash ./scripts/train_one_node.sh
You can monitor training progress by checking the log files in director "./log"