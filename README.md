# GRIT: Teaching MLLMs to Think with Images

**Grounded Reasoning wiht Texts and Images (GRIT)** is a novel method for training Multimodal Large Language Models (MLLMs) to perform grounded reasoning by generating reasoning chains that interleave natural language and explicit bounding box coordinates. This approach can use as few as **20 training data samples** to enable models to ground their reasoning in specific image regions, achieving a **unified grounding and reasoning ability**.

<div align="center">
  <a href="https://arxiv.org/abs/2505.15879">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square" alt="Paper">
  </a>
  <a href="https://grounded-reasoning.github.io">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square" alt="Project Page">
  </a>
  <a href="https://b86dd615e41b242e22.gradio.live">
    <img src="https://img.shields.io/badge/Live-Demo-green?style=flat-square" alt="Live Demo">
  </a>
</div>

## Examples of GRIT's Grounded Reasoning

<img src="readme_images/eg1.png" alt="Example 1" width="800">

<details>
<summary>More examples (click to expand)</summary>

<img src="readme_images/eg2.png" alt="Example 2" width="800">

<img src="readme_images/eg3.png" alt="Example 3" width="800">

</details>

## Pretrained Models

Pretrained GRIT models are available on Hugging Face:

- [GRIT-20-InternVL-2B](https://huggingface.co/yfan1997/GRIT-20-InternVL-2B): GRIT model based on InternVL 2B
- [GRIT-20-Qwen2.5-VL-3B](https://huggingface.co/yfan1997/GRIT-20-Qwen2.5-VL-3B): GRIT model based on Qwen2.5-VL 3B

## Setup

1. Optional conda environment install
    ```bash
    conda create -n gprogr python=3.12
    ```
2. **Clone the repository with submodules:**
   ```bash
   git clone --recurse-submodules https://github.com/UeFan/GRIT.git
   cd GRIT
   ```
3. **Run the setup script:**
   ```bash
   bash setup.sh
   ```
4. **Set up Weights & Biases for experiment tracking:**
   ```bash
   pip install wandb
   wandb login
   ```

5. **Set up your OpenAI credentials:**
   Create a file named `gpt_credentials.py` with the following content:
   ```python
   api_base = ""
   api_key = ""
   deployment_name = ""
   api_version = ""
   ```
   Fill in your credentials as needed for API access.

6. **Download data from Hugging Face:**
   ```bash
   git lfs install
   git clone https://huggingface.co/datasets/yfan1997/GRIT_data
   ```
   Follow the instructions in GRIT_data/README.md to download image data and place it within the GRIT_data directory.

## Training and evaluation

### Training

To train models with GRIT using the grounded reasoning approach:

#### InternVL
```bash
bash scripts/8_80gpu_20_train_internvl_grounded_reasoning_single_turn_think_rethink.sh
```

#### Qwen
```bash
bash scripts/8_80gpu_20_train_qwen_grounded_reasoning_single_turn_think_rethink.sh
```

### Evaluation

#### Using training scripts for evaluation
To evaluate models instead of training, add the following parameters to any training script:
```bash
--num_train_epochs 0
--eval_on_start True 
--model_name_or_path MODEL_NAME
```

Replace `MODEL_NAME` with the path to your trained model checkpoint.


More evaluation scripts are available in scripts/


## Citation

```
@misc{fan2025gritteachingmllmsthink,
      title={GRIT: Teaching MLLMs to Think with Images}, 
      author={Yue Fan and Xuehai He and Diji Yang and Kaizhi Zheng and Ching-Chen Kuo and Yuting Zheng and Sravana Jyothi Narayanaraju and Xinze Guan and Xin Eric Wang},
      year={2025},
      eprint={2505.15879},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.15879}, 
}
``` 