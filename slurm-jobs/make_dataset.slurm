#! /bin/sh

#SBATCH --job-name=mk-data
#SBATCH --output=logs/mk-data.out # redirect stdout
#SBATCH --error=logs/mk-data.err # redirect stderr
#SBATCH --partition=gpu-sharifm   #killable #gpu-a100-killable  #killable  #studentbatch # (see resources section)
#SBATCH --time=1300 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=30000 # CPU memory (MB)
#SBATCH --gpus=1

export PYTHONPATH="$PWD"  # set the Python's path to the current path
export HF_HOME="/home/sharifm/students/matanbentov"  # modify to home de-facto dir

DATASET=coco_captions  # coco_captions, conc_captions, nq-corpus
EMB_MODEL="intfloat/e5-base-v2"  #  sentence-transformers/gtr-t5-base, intfloat/e5-base-v2, openai/clip-vit-large-patch14
BATCH_SIZE=512

python cli.py create-dataset ${DATASET} ${EMB_MODEL}  \
    --batch-size ${BATCH_SIZE}