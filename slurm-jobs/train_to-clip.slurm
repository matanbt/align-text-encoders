#! /bin/sh

#SBATCH --job-name=tr-aligner
#SBATCH --output=logs/tr-aligner.out # redirect stdout
#SBATCH --error=logs/tr-aligner.err # redirect stderr
#SBATCH --partition=gpu-sharifm   #killable #gpu-a100-killable  #killable  #studentbatch # (see resources section)
#SBATCH --time=3000 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=30000 # CPU memory (MB)
#SBATCH --gpus=1

export PYTHONPATH="$PWD"  # set the Python's path to the current path
export HF_HOME="/home/sharifm/students/matanbentov"  # modify to home de-facto dir


# >>>>>>>>>>>> Model Choice:
### LINEAR
ALIGNER_TYPE="mlp"
HPARAMS="--n-hidden-layers 0"


### MLP
#ALIGNER_TYPE="mlp"
#HPARAMS="--n-hidden-layers 3"

### TRANSFORMER
# ALIGNER_TYPE="transformer"
# HPARAMS="--num-blocks 4"


# >>>>>>>>>>>> Align multi-modal
### COCO:
# DATASET=coco_captions
### CONC:
DATASET=conc_captions
###
TARGET_MODEL="openai/clip-vit-large-patch14"
SOURCE_MODEL="intfloat/e5-base-v2"  # sentence-transformers/average_word_embeddings_glove.6B.300d
EVAL_ON="cifar100"
BATCH_SIZE=400_000  # 100_000 for `killable`
OUT_DIR=out/clip-to-e5--linear4


# >>>>>>>>>>>> ALIGN:
# run training
python cli.py train ${DATASET} ${SOURCE_MODEL} ${TARGET_MODEL}  \
    --aligner-type ${ALIGNER_TYPE} --eval-on ${EVAL_ON}  \
    ${HPARAMS} \
    --batch-size ${BATCH_SIZE} --out-dir ${OUT_DIR} --n-epochs 150 --learning-rate 0.05


# >>>>>>>>>>>> EVAL:
# Classification:
clip_benchmark eval --dataset cifar10 cifar100 imagenet1k --task zeroshot_classification --model source source+aligner target --pretrained NONE \
  --model_type our_experimental_models  --model_cache_dir "${OUT_DIR}"  \
  --output "${OUT_DIR}/benchmark_{dataset}_{model}_{task}.json" --batch_size 1024

# Retrieval: # [mscoco_captions, flickr8k,  flickr30k]
clip_benchmark eval --dataset wds/flickr8k wds/flickr30k wds/mscoco_captions \
  --task zeroshot_retrieval --model source source+aligner target --pretrained NONE \
  --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
  --model_type our_experimental_models  --model_cache_dir ${OUT_DIR}  \
   --output "${OUT_DIR}/benchmark_{dataset}_{model}_{task}.json" --batch_size 1024
