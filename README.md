# Aligning Representations Across Different Text Encoders

![flow.png](flow.png)

[[Report]](https://matanbt.github.io/post/align2024/report.pdf) [[Blogpost]](https://matanbt.github.io/post/align2024/)


This repository contains code for mapping embedding space of one text encoder to another.
We evaluate these mapping ("aligners") for different applications, including:
mapping unimodal text encoder to [CLIP](https://arxiv.org/abs/2103.00020)'s text encoder and evaluating on common multimodal tasks;
mapping text encoders to an embedding space invertible by [Vec2Text](https://arxiv.org/abs/2310.06816) and inverting their embeddings to text.

This project was done as part of the course _Advanced Topics in Deep Learning_, TAU.

## Installation

Run on isolated Python 3.8+ environment, and ensure submodules are updated.

Install the forked packages:
```bash
>> cd CLIP_benchmark
>> python setup.py install

>> cd vec2text
>> python setup.py install
```

Next, install the original CLIP-benchmark repository:
```bash
>> cd CLIP_benchmark
>> pip install -e .
```

## Usage

### Training
Training an aligner requires: 
1. Creating datasets for _source_ and _target_ text-encoders embeddings; available via `slurm-jobs/make_dataset.slurm`.
2. training the aligner; available via `slurm-jobs/train_to-{clip,text}.slurm`.


### Evaluation
The following describes how to evaluate an existing aligner, located in `./out/{aligner_dir}/`, on different tasks.

#### Multimodal benchmarks - Image Classification
```bash
clip_benchmark eval --dataset cifar10 cifar100 imagenet1k --task zeroshot_classification --model source source+aligner target --pretrained NONE \
  --model_type our_experimental_models  --model_cache_dir "${OUT_DIR}"  \
  --output "${OUT_DIR}/benchmark_{dataset}_{model}_{task}.json" --batch_size 1024
```

#### Multimodal benchmarks - Text-Image Retrieval
```bash
clip_benchmark eval --dataset wds/flickr8k wds/flickr30k wds/mscoco_captions \
  --task zeroshot_retrieval --model source+aligner target --pretrained NONE \
  --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
  --model_type our_experimental_models  --model_cache_dir ${OUT_DIR}  \
   --output "${OUT_DIR}/benchmark_{dataset}_{model}_{task}.json" --batch_size 1024
```

#### Text Inversion with Vec2Text
```bash
python cli.py evaluate text_inversion__nq ${OUT_DIR}
```


