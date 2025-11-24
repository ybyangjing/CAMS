# CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement

* **Title**: **[CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement](https://arxiv.org/pdf/2511.16378)**
* **Authors**: **Pan Yang**, **Cheng Deng**, **Jing Yang**, **Han Zhao**, **Yun Liu**, **Yuling Chen**, **Xiaoli Ruan**, **Yanping Chen**
* **Institutes**: Guizhou University, Shanghai Jiao Tong University, Nankai University

## Overview
<p align="center">
  <img src="https://github.com/ybyangjing/CAMS/blob/main/img/framework.png">
</p>

Compositional zero-shot learning (CZSL) aims to recognize unseen attribute–object compositions from seen ones, but CLIP-based methods often suffer from limited disentanglement due to the restricted capacity of global representations. We propose CAMS, a semantic disentangled framework that leverages high-level semantic features, employs Gated Cross-Attention to adaptively extract fine-grained visual features, and applies Multi-Space Disentanglement to align semantic features with prompts, improving generalization to unseen compositions. Evaluated on MIT-States, UT-Zappos, and C-GQA, CAMS achieves state-of-the-art performance, boosting HM by up to **+9.3%** and AUC by **+12.4%** in both closed-world and open-world settings.

## Environment 
The model code is implemented based on the PyTorch framework. The experimental environment includes:

- Ubuntu20.04
- Intel(R) Core(TM) i9-12900K CPU
- 128GB RAM
- NVIDIA RTX 6000 GPU

## Dataset
We experiment with three datasets: MIT-States, UT-Zappos, and C-GQA.
The three datasets can be downloaded by clicking [here](https://pan.baidu.com/s/1EjuOntesOsOlE26wpu2kAg?pwd=4g4b).

You only need to modify the **config file path** and the corresponding **dataset_path** in ``CAMS\flags.py``.
## Results
### Main Results

The following results show the closed-world and open-world performance of CAMS, compared with state-of-the-art methods using the same backbone (ViT-L/14). Additional experimental results can be found in the paper.
<p align="center">
  <img src="https://github.com/ybyangjing/CAMS/blob/main/img/main_results.png" alt="main_results">
</p>

## Acknowledgement
Our code references the following project:
* [Troika](https://github.com/bighuang624/Troika)

## Contact
**If you have any questions you can contact us : gs.pyang24@gzu.edu.cn or jyang23@gzu.edu.cn**
