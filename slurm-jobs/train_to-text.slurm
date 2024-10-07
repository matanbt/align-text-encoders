#! /bin/sh

#SBATCH --job-name=tr-aligner
#SBATCH --output=logs/tr-aligner.out # redirect stdout
#SBATCH --error=logs/tr-aligner.err # redirect stderr
#SBATCH --partition=gpu-sharifm   #killable #gpu-a100-killable  #killable  #studentbatch # (see resources section)
#SBATCH --nodelist=n-602  # n-{602,804}
#SBATCH --time=2500 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --gpus=1

export PYTHONPATH="$PWD"  # set the Python's path to the current path
export HF_HOME="/home/sharifm/students/matanbentov"  # modify to home de-facto dir


# >>>>>>>>>>>> Model Choice:
### LINEAR
ALIGNER_TYPE="mlp"
HPARAMS="--n-hidden-layers 0"
LR=0.01
BATCH_SIZE=250_000

### MLP
#ALIGNER_TYPE="mlp"
#HPARAMS="--n-hidden-layers 3"
#LR=0.002
#BATCH_SIZE=250_000

### TRANSFORMER
# ALIGNER_TYPE="transformer"
# HPARAMS="--num-blocks 4"
# LR=0.0001
# BATCH_SIZE=100_000


# >>>>>>>>>>>> Align for inversion
DATASET=nq-corpus
###
TARGET_MODEL="sentence-transformers/gtr-t5-base"
SOURCE_MODEL="intfloat/e5-base-v2"
EVAL_ON="text_inversion"
BATCH_SIZE=400_000
OUT_DIR=out/gtr-to-e5--linear21

### Cross embedding
TARGET_MODEL="sentence-transformers/all-MiniLM-L6-v2" # align to
SOURCE_MODEL="intfloat/e5-base-v2"  # use this model to attack
EVAL_ON="text_inversion"
OUT_DIR=out/minilm-to-e5--linear

# >>>>>>>>>>>> ALIGN:
# run training
python cli.py train ${DATASET} ${SOURCE_MODEL} ${TARGET_MODEL}  \
    --aligner-type ${ALIGNER_TYPE} --eval-on ${EVAL_ON}  \
    ${HPARAMS} \
    --batch-size ${BATCH_SIZE} --out-dir ${OUT_DIR} --n-epochs 500 --learning-rate ${LR}

# >>>>>>>>>>>> EVAL:
python cli.py evaluate text_inversion__nq ${OUT_DIR}
python cli.py evaluate text_inversion__msmarco ${OUT_DIR}
