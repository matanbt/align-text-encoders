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
# >> TARGETS: sentence-transformers/gtr-t5-base, openai/clip-vit-large-patch14
# >> SOURCES: intfloat/e5-base-v2, sentence-transformers/average_word_embeddings_glove.6B.300d, sentence-transformers/all-MiniLM-L6-v2, nomic-ai/nomic-embed-text-v1, Snowflake/snowflake-arctic-embed-m
EMB_MODEL="intfloat/e5-base-v2"
BATCH_SIZE=512

python cli.py create-dataset ${DATASET} ${EMB_MODEL}  \
    --batch-size ${BATCH_SIZE}