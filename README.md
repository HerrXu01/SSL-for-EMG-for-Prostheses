# Self-Supervised Feature Learning in EMG Data for Upper-Limb Prostheses

## Introduction

This repository contains the official implementation of the master's thesis *"Self-Supervised Feature Learning in EMG Data for Upper-Limb Prostheses"* at the Technical University of Munich.

We propose a self-supervised learning framework that leverages pretrained large language models (LLMs), specifically LLaMA-7B, to extract meaningful features from EMG signals without requiring manual labels. By formulating EMG sequences as pseudo-language and training the model via next-token prediction, our method learns robust representations that improve gesture classification performance, even with limited labeled data.

Experiments on the Ninapro DB5 dataset demonstrate strong label efficiency and cross-gesture/subject generalization, showing the potential of LLMs as universal sequence learners for biosignal modeling.

## Installation

### 1. Create Conda Environment

We recommend using a dedicated Conda environment to ensure compatibility and reproducibility:

```bash
conda create -n emg-ssl python=3.10 -y
conda activate emg-ssl
```

### 2. Install Dependencies

Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Prepare LLaMA-2-7B Model

Download the pretrained **LLaMA-2-7B** model from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf).
Create a folder named `llama` in the project root, and place the downloaded model files inside.
Make sure the folder contains **two `.safetensors` files** (typically the model weights).

Your folder structure should look like this:

```
project_root/
├── llama/
│   ├── config.json
│   ├── tokenizer.model
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   └── ...
```

## Usage

### 1. Preprocessing (Only if Required)

If the directory `dataset/timestamp_embeddings/` is **empty**, run all preprocessing scripts located in `scripts/preprocess/`:

```bash
sh scripts/preprocess/sl80_tl4_sr200.sh 
sh scripts/preprocess/sl80_tl8_sr200.sh 
...
```

If the directory is **not empty**, you can skip this step.

### 2. Run Experiments

#### Phase 1: SSL Pretraining and In-Distribution Evaluation (e.g., Sub-Experiment 1)

```bash
# Step 1: Self-supervised pretraining
sh scripts/phase_1/exp_1_sl80_tl4_mixTrue/p1_e1_01_pretrain.sh

# Step 2: Feature extraction using the trained SSL model
sh scripts/phase_1/exp_1_sl80_tl4_mixTrue/p1_e1_02_extract.sh

# Step 3 (optional): Train CNN classifier (deprecated)
sh scripts/phase_1/exp_1_sl80_tl4_mixTrue/p1_e1_03_ssl_classifier.sh

# Step 4: Train MLP classifier on extracted features (recommended)
sh scripts/phase_1/exp_1_sl80_tl4_mixTrue/p1_e1_04_ssl_mlp_classifier.sh
```

#### Phase 2: SSL Classifiers vs. Supervised Classifiers, under in-distribution and transfer setting (e.g., Experiment 2)

```bash
# Step 1: Extract features for new gestures using a pretrained extractor (e.g., from Phase 1, Exp 16)
sh scripts/phase_2/exp_2_gesture_transfer_mlp/p2_e2_01_extract_features_on_new_gestures_use_p1e16_extractor.sh

# Step 2: Train MLP classifier using SSL features
sh scripts/phase_2/exp_2_gesture_transfer_mlp/p2_e2_02_ssl_mlp_classifier_based_on_p1e16.sh

# Step 3: Train MLP classifier in fully supervised manner (baseline)
sh scripts/phase_2/exp_2_gesture_transfer_mlp/p2_e2_03_supervised_mlp_classifier_based_on_p1e16.sh
```
