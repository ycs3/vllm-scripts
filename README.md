# Temporal Grounding of Activities using Multimodal Large Language Models

Temporal activity localization using a two-stage approach: Stage 1) multimodal LLM, Stage 2) text-based LLM

* [Paper](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/YoungCholSong.pdf): Temporal Grounding of Activities using Multimodal Large Language Models
* [Scripts](https://github.com/ycs3/vllm-scripts/tree/main/scripts): Scripts for finetuning, querying APIs and running vision and text-only LLMs.

## Temporal Grounding Example
Comparing different model outputs of temporal activity localization:

<img src="https://github.com/ycs3/vllm-scripts/assets/30537892/cae7268d-3607-4249-b643-884ce16946f7" width="500" alt="image">

## Instruction Tuning Results
R@IoU for base LLaVA, instruction-tuned LLaVA using the general prompting strategy:

<img src="https://github.com/ycs3/vllm-scripts/assets/30537892/9b27cfc3-1987-4d0c-801c-037e98dc77a8" width="500" alt="image">

## Two-stage LLM Results
Comparing metrics (IoU) for different multimodal and text-based LLMs:

<img src="https://github.com/ycs3/vllm-scripts/assets/30537892/098cb8ed-9fcb-4418-b55e-4fa49778d6bf" width="451" alt="image">
