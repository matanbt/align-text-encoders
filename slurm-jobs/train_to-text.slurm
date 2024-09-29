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


# >>>>>>>>>>>> Align for inversion
DATASET=nq-corpus
###
TARGET_MODEL="sentence-transformers/gtr-t5-base"
SOURCE_MODEL="intfloat/e5-base-v2"
EVAL_ON="text_inversion"
BATCH_SIZE=400_000
OUT_DIR=out/gtr-to-e5--linear21


# >>>>>>>>>>>> ALIGN:
# run training
python cli.py train ${DATASET} ${SOURCE_MODEL} ${TARGET_MODEL}  \
    --aligner-type ${ALIGNER_TYPE} --eval-on ${EVAL_ON}  \
    ${HPARAMS} \
    --batch-size ${BATCH_SIZE} --out-dir ${OUT_DIR} --n-epochs 150 --learning-rate 0.05

# >>>>>>>>>>>> EVAL:
python cli.py evaluate text_inversion__nq ${OUT_DIR}
python cli.py evaluate text_inversion__msmarco ${OUT_DIR}
