# COCO Caption Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/coco_caption_classification/blob/main/notebooks/train_colab.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_SPACE)

This project focuses on **Multimodal Classification** using the COCO dataset. It evaluates the model’s ability to align visual semantics with complex linguistic context through **Zero-shot** and **Few-shot** learning.

Instead of standard categorical classification, the model is presented with an image and a set of $N$ candidate text descriptions ($N \ge 5$). The objective is to identify the one true ground-truth description, overcoming the $N-1$ distractors.

---

## 📂 Project Structure

```text
coco_caption_classification/
├── coco_subset_images/              # Extracted image data + JSON
│   └── coco_multimodal_subset.json  # Base metadata definition
├── models/                          # Weights & checkpoints
├── notebooks/
│   └── train_colab.ipynb            # Google Colab main training notebook
├── src/                             # Shared source scripts
│   ├── data_loader.py               # Dataset objects, subset loading and splits
│   ├── model_arch.py                # Wrapper models based on openai/CLIP
│   ├── Image_ID_Exclusive_Sampling.ipynb # The data subset generation script
│   └── utils.py                     # Evaluation functions & parameter counter
├── github_page/                     # Visual presentation output
├── settings/
│   ├── requirements.txt             # Primary environment targets
│   ├── requirements_gradio.txt      # Huggingface targeted deps
│   └── config.yaml                  # Model and wandb variables
├── app.py                           # Gradio Hugging Face Application
└── README.md                        # Documentation
```

## 🚀 How to Run

### Development & Training

1. Clone the project to your drive/local machine.
2. Upload the `coco_multimodal_subset.json` and the corresponding images to `coco_subset_images/`.
3. Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Colab.
4. Mount your drive and install dependencies from `settings/requirements.txt`.
5. Run the cells to train. The configuration can be modified in `settings/config.yaml`.

### Gradio Inference Demo

To launch the interactive N-shot visualizer app:

```bash
pip install -r settings/requirements_gradio.txt
python app.py
```

This renders a randomized subset display mapping 1 image against N distractors and graphs expected performance tracking across model backbones (ViT vs. RN50).

## 🔬 Core Workflow

1. **Vision Extractor:** Converts inputs using normalizations standard to CLIP variants (`ViT-B/32` or `RN50`).
2. **Text Extractor:** Standardizes $N$-size list of texts into embedding tensors.
3. **Linear Classification Probing:** Employs Cross-Entropy across dot-product similarities spanning $k=\{0, 8, 16, 32\}$.
4. **Analysis:** Metrics (Acc, Macro-F1), inference compute sizes, and GradCAM/EigenCam map generations.

## 📊 Report and Info

Please refer to the [GitHub Pages Site](./github_page/index.html) or the provided Youtube Links for the complete metric visualization and architectural diagrams breakdown.
