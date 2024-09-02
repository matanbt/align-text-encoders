#! /bin/sh

#SBATCH --job-name=eval
#SBATCH --output=logs/eval.out # redirect stdout
#SBATCH --error=logs/eval.err # redirect stderr
#SBATCH --partition=gpu-sharifm   #killable #gpu-a100-killable  #killable  #studentbatch # (see resources section)
#SBATCH --time=1300 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=30000 # CPU memory (MB)
#SBATCH --gpus=1

export PYTHONPATH="$PWD"  # set the Python's path to the current path
export HF_HOME="/home/sharifm/students/matanbentov"  # modify to home de-facto dir



OUT_DIR="out/clip-to-e5--linear5/"
clip_benchmark eval --dataset cifar10 cifar100 imagenet1k --task zeroshot_classification --model source source+aligner target --pretrained NONE \
  --model_type our_experimental_models  --model_cache_dir out/clip-to-e5--linear5/  \
   --output "${OUT_DIR}/benchmark_{dataset}_{model}_{task}.json" --batch_size 1024

# QUICK EVALUATION (internal impl.)
#TASK_NAME=cifar100
#python cli.py evaluate ${TASK_NAME} ${OUT_DIR}
