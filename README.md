# DEEPLEARNING AI Courses

A collection of notes on a course from [DeepLearning.AI](https://learn.deeplearning.ai/) that I have taken and created notes from.

## Courses List

| No. | Course Name                              | Finished |
|-----|------------------------------------------|----------|
| 1   | Intro to Federated Learning              | ❌       |
| 2   | Open Source Models with Hugging Face     | ❌       |
| 3   | Prompt Engineering with Llama            | ❌       |
| 4   | Quantization Fundamentals with Hugging Face | ✅       |
| 5   | Quantization in Depth                    | ✅       |
| 6   | Introduction to on-device AI             | ✅       |

## Getting Started

### Huggingface Setup

1. Install PyTorch or TensorFlow with GPU CUDA support.
2. Install `transformers` by typing `pip install transformers`.
3. Install HuggingFace CLI by typing `pip install -U "huggingface_hub[cli]"`.
4. Set up an access token on the HuggingFace platform.
5. Log in to HuggingFace CLI by typing `huggingface-cli login`.
   - If you do not want to be prompted, type `huggingface-cli login --token $HF_TOKEN --add-to-git-credential`.
   - You need to pass your token as an environment variable.
6. Verify the HuggingFace login by typing `huggingface-cli whoami`.
7. Download the HuggingFace repository by typing `huggingface-cli download [name-of-repository]`.
