# TOCSIN
**This code is for paper "Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness"**, where we borrow code and data from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt).

## Data
Following folders are created for experiments:
* ./exp_Open_source_model -> experiments for open-source models generations (Five_models.sh).
* ./exp_API-based_model -> experiments for ChatGPT, GPT-4, and Gemini generations (API-based.sh).

## Models loading
If you want to load models locally, place the files for the bart-base model in the 'facebook' directory. 

For experiments with Open-Source LLMs, Please download models and create directories in the following format:
```
gpt2-xl: './gpt2-xl'
```
```
opt-2.7b: 'facebook/opt-2.7b'
```  
```
gpt-neo-2.7B: 'EleutherAI/gpt-neo-2.7B'
```
```
gpt-j-6B: 'EleutherAI/gpt-j-6B'
```
```
gpt-neox-20b: 'EleutherAI/gpt-neox-20b'
```

## Environment
* Python3.8
* PyTorch2.1.0

GPU: NVIDIA A40 GPU with 48GB memory

## Demo
Please run following commands for a demo:
```
sh Five_models.sh
```
for experiments with Open-Source LLMs
or
```
sh API-based.sh
```
for experiments with API-based LLMs

### Citation
If you find this work useful, you can cite it with the following BibTex entry:



