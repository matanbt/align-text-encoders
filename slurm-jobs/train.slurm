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

DATASET=coco_captions
TARGET_MODEL="sentence-transformers/clip-ViT-L-14"
SOURCE_MODEL="intfloat/e5-base-v2"
N_HIDDEN_LAYERS=0  # 0 for linear projection
EVAL_ON="cifar100"
BATCH_SIZE=11_000
python cli.py train ${DATASET} ${SOURCE_MODEL} ${TARGET_MODEL} ${N_HIDDEN_LAYERS} ${EVAL_ON}  \
    --batch-size ${BATCH_SIZE} --out-dir out/clip-to-e5--linear1
