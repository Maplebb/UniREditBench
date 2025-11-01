<div align="center">
    <h1 align="center">UniREditBench: A Unified Reasoning-based Image Editing Benchmark
    </h1>


[UnifiedReward](https://github.com/CodeGoat24/UnifiedReward) Team

Shanghai Innovation Institue

<a href="">
<img src='https://img.shields.io/badge/arXiv-UniREditBench-blue' alt='Paper PDF'></a>







<a href="https://maplebb.github.io/UniREditBench/">
<img src='https://img.shields.io/badge/Website-UniREditBench-orange' alt='Project Page'></a>




[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-UniREdit_Data_100K-yellow)](https://huggingface.co/datasets/CodeGoat24/UniGenBench-Eval-Images)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-UniREdit_Bagel-yellow)](https://huggingface.co/CodeGoat24/UniGenBench-EvalModel-qwen-72b-v1) 
</div>


## 🔥 News
- [2025/11] 🔥🔥 We release [project page](https://maplebb.github.io/UniREditBench/) of UniREditBench!!




## Introduction

We propose <b>UniREditBench</b>, a unified benchmark for reasoning-based image editing assessment with broader evaluation dimension coverage and robust evaluation pipeline. We also design an automated multi-scenario data synthesis pipeline and construct <b>UniREdit-Data-100K</b>, a large-scale synthetic dataset with high-quality chain-of-thought (CoT) reasoning annotations. We fine-tune Bagel on this dataset and develop <b>UniREdit-Bagel</b>, demonstrating substantial improvements in both in-domain and out-of-distribution settings.


<img   alt="image" src="docs/static/images/teaser.png" />

<img alt="image" src="docs/static/images/radar.png" />

### ✨ Highlights:

- **Broader Scenario and Reasoning Dimension Coverage**:  It contains 2,700 high-quality samples organized into 8 primary reasoning dimensions and 18 sub-categories, spanning both real-world and game-world image editing tasks.



- **Reliable Dual-Reference Evaluation.**: For each sample assessment, we design both the textual reference and ground-truth (GT) image reference. This multi-modal reference enables vision-language model (VLM) evaluators to perform direct and fine-grained comparisons at both the textual and visual levels with the generated images, leading to more reliable evaluation.


<img alt="image" src="docs/static/images/motivation_tab.png" />
<img alt="image" src="docs/static/images/motivation_fig.png" />

<img alt="image" src="docs/static/images/testpoint_cases.png" />


## 📧 Contact
If you have any comments or questions, please open a new issue or feel free to contact [Feng Han](fhan25@m.fudan.edu.cn) and [Yibin Wang](https://codegoat24.github.io).


## ⭐ Citation
```bibtex
```

