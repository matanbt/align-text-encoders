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

# >>>>>>>>>>>> Model Choice:
### LINEAR
ALIGNER_TYPE="mlp"
N_HIDDEN_LAYERS=0  # 0 for linear projection

### MLP
#ALIGNER_TYPE="mlp"
#N_HIDDEN_LAYERS=3  # 0 for linear projection

### TRANSFORMER
#ALIGNER_TYPE="transformer"
#NUM_BLOCKS=4

# >>>>>>>>>>>> GOAL 1: Align multi-modal
### COCO:
# DATASET=coco_captions
### CONC:
DATASET=conc_captions
###
TARGET_MODEL="openai/clip-vit-large-patch14"
SOURCE_MODEL="intfloat/e5-base-v2"
EVAL_ON="cifar100"
BATCH_SIZE=100_000
OUT_DIR=out/clip-to-e5--linear4


# >>>>>>>>>>>> GOAL 2: Align for inversion
DATASET=nq-corpus
###
TARGET_MODEL="sentence-transformers/gtr-t5-base"
SOURCE_MODEL="intfloat/e5-base-v2"
EVAL_ON="text_inversion"
BATCH_SIZE=100_000
OUT_DIR=out/gtr-to-e5--linear21


# >>>>>>>>>>>> ALIGN:
# run training
python cli.py train ${DATASET} ${SOURCE_MODEL} ${TARGET_MODEL} ${N_HIDDEN_LAYERS} ${EVAL_ON}  \
    --batch-size ${BATCH_SIZE} --out-dir ${OUT_DIR} --n-epochs 150 --learning-rate 0.05

# run eval.slurm [TODO]
