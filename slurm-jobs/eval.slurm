#! /bin/sh

#SBATCH --job-name=eval
#SBATCH --output=logs/eval.out # redirect stdout
#SBATCH --error=logs/eval.err # redirect stderr
#SBATCH --partition=gpu-sharifm   #killable #gpu-a100-killable  #killable  #studentbatch # (see resources section)
#SBATCH --nodelist=n-602  # n-{602,804}
#SBATCH --time=3000 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=30000 # CPU memory (MB)
#SBATCH --gpus=1

export PYTHONPATH="$PWD"  # set the Python's path to the current path
export HF_HOME="/home/sharifm/students/matanbentov"  # modify to home de-facto dir


# >>> Multi-Modal Eval: [follows https://github.com/LAION-AI/CLIP_benchmark/tree/main/benchmark#webdataset-evaluation-vtab-and-retrieval-datasets-mscoco-flickr8k-flickr30k]
OUT_DIR="out/clip-to-e5--linear/"
# Classification:
  # >> ?? wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
  # `source` cannot be evaluated if it's not the same dimensions!
  # evaluated datasets: `imagenet1k cifar10 cifar100`
clip_benchmark eval --dataset cifar10 cifar100 imagenet1k --task zeroshot_classification --model source source+aligner target --pretrained NONE \
  --model_type our_experimental_models  --model_cache_dir "${OUT_DIR}"  \
  --output "${OUT_DIR}/benchmark_{dataset}_{model}_{task}.json" --batch_size 1024

# Classification via webdatasets:
# clip_benchmark eval --dataset wds/imagenet1k wds/vtab/cifar10 wds/vtab/cifar100 --task zeroshot_classification --model source source+aligner target --pretrained NONE \
#    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
#    --model_type our_experimental_models  --model_cache_dir "${OUT_DIR}"  \
#    --output "${OUT_DIR}/benchmark_{dataset}_{model}_{task}.json" --batch_size 1024

# Retrieval: # [mscoco_captions, crossmodal3600, flickr8k,  flickr30k]
clip_benchmark eval --dataset wds/flickr8k wds/flickr30k wds/mscoco_captions \
  --task zeroshot_retrieval --model source+aligner target --pretrained NONE \
  --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
  --model_type our_experimental_models  --model_cache_dir ${OUT_DIR}  \
   --output "${OUT_DIR}/benchmark_{dataset}_{model}_{task}.json" --batch_size 1024

# Quick eval (internal impl.) [DISABLED]
# python cli.py evaluate cifar100 ${OUT_DIR}

# --------------------------------------------------------------------------------
# >> Vec2Text Eval:
OUT_DIR="out/gtr-to-e5--linear/"
python cli.py evaluate text_inversion__nq ${OUT_DIR}
#python cli.py evaluate text_inversion__msmarco ${OUT_DIR}
