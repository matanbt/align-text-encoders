#! /bin/sh

#SBATCH --job-name=tr-aligner
#SBATCH --output=logs/tr-aligner.out # redirect stdout
#SBATCH --error=logs/tr-aligner.err # redirect stderr
#SBATCH --partition=killable   #killable #gpu-a100-killable  #killable  #studentbatch # (see resources section)
#SBATCH --time=1300 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=30000 # CPU memory (MB)
#SBATCH --gpus=1

export PYTHONPATH="$PWD"  # set the Python's path to the current path
export HF_HOME="/home/sharifm/students/matanbentov"  # modify to home de-facto dir

# >>>>>>>>>>>> GOAL 1: Align multi-modal
### COCO:
# DATASET=coco_captions
### CONC:
DATASET=conc_captions
###
TARGET_MODEL="openai/clip-vit-large-patch14"
SOURCE_MODEL="intfloat/e5-base-v2"
N_HIDDEN_LAYERS=0  # 0 for linear projection
EVAL_ON="cifar100"
BATCH_SIZE=100_000
OUT_DIR=out/clip-to-e5--linear4

# >>>>> GOAL 2: Align for inversion
DATASET=nq-corpus
###
TARGET_MODEL="sentence-transformers/gtr-t5-base"
SOURCE_MODEL="intfloat/e5-base-v2"
N_HIDDEN_LAYERS=0  # 0 for linear projection
EVAL_ON="text_inversion"
BATCH_SIZE=100_000
OUT_DIR=out/gtr-to-e5--linear4

# run training
python cli.py train ${DATASET} ${SOURCE_MODEL} ${TARGET_MODEL} ${N_HIDDEN_LAYERS} ${EVAL_ON}  \
    --batch-size ${BATCH_SIZE} --out-dir ${OUT_DIR} --n-epochs 150 --learning-rate 0.05

# run benchmark
clip_benchmark eval --dataset cifar10 cifar100 imagenet1k --task zeroshot_classification \
  --model source source+aligner target --pretrained NONE \
  --model_type our_experimental_models  --model_cache_dir out/clip-to-e5--linear5/  \
  --output "${OUT_DIR}/benchmark_{dataset}_{model}_{task}.json" --batch_size 1024