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

TASK_NAME=cifar100
OUT_DIR="out/clip-to-e5--linear2/"
python cli.py evaluate ${TASK_NAME} ${OUT_DIR}
