# DEEPLEARNING AI Courses

A collection of notes on a course from [DeepLearning.AI](https://learn.deeplearning.ai/) that I have taken and created notes from.

## Courses List

### Sorted Courses List

| No. | Course Name                              | Finished |
|-----|------------------------------------------|----------|
| 1   | Attention in Transformers                | ❌       |
| 2   | Intro to Federated Learning              | ❌       |
| 3   | Introduction to on-device AI             | ✅       |
| 4   | Open Source Models with Hugging Face     | ❌       |
| 5   | Prompt Engineering with Llama            | ❌       |
| 6   | Quantization Fundamentals with Hugging Face | ✅       |
| 7   | Quantization in Depth                    | ✅       |
| 8   | Reasoning with O1 and O2                 | ✅       |

## Getting Started

### Environment Setup

1. Download Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
2. Install Miniconda by following the instructions.
3. Create a new environment by typing `conda create -n deeplearningai python=3.8` or `python -m venv venv`.
4. Activate the environment by typing `conda activate deeplearningai` or `.\venv\Scripts\Activate`.

### Huggingface Setup

1. Install PyTorch or TensorFlow with GPU CUDA support.
   
   ```powershell
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
   ```
   
2. Install `transformers` by typing `pip install transformers`.
3. Install HuggingFace CLI by typing `pip install -U "huggingface_hub[cli]"`.
4. Set up an access token on the HuggingFace platform.
5. Log in to HuggingFace CLI by typing `huggingface-cli login`.
   - If you do not want to be prompted, type `huggingface-cli login --token $HF_TOKEN --add-to-git-credential`.
   - You need to pass your token as an environment variable.
6. Verify the HuggingFace login by typing `huggingface-cli whoami`.
7. Download the HuggingFace repository by typing `huggingface-cli download [name-of-repository]`.
