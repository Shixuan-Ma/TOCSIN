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

    @inproceedings{ma-wang-2024-zero,
    title = "Zero-Shot Detection of {LLM}-Generated Text using Token Cohesiveness",
    author = "Ma, Shixuan  and
      Wang, Quan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.971/",
    doi = "10.18653/v1/2024.emnlp-main.971",
    pages = "17538--17553",
    abstract = "The increasing capability and widespread usage of large language models (LLMs) highlight the desirability of automatic detection of LLM-generated text. Zero-shot detectors, due to their training-free nature, have received considerable attention and notable success. In this paper, we identify a new feature, token cohesiveness, that is useful for zero-shot detection, and we demonstrate that LLM-generated text tends to exhibit higher token cohesiveness than human-written text. Based on this observation, we devise TOCSIN, a generic dual-channel detection paradigm that uses token cohesiveness as a plug-and-play module to improve existing zero-shot detectors. To calculate token cohesiveness, TOCSIN only requires a few rounds of random token deletion and semantic difference measurement, making it particularly suitable for a practical black-box setting where the source model used for generation is not accessible. Extensive experiments with four state-of-the-art base detectors on various datasets, source models, and evaluation settings demonstrate the effectiveness and generality of the proposed approach. Code available at: https://github.com/Shixuan-Ma/TOCSIN."
}

