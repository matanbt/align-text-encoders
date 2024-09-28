# align-text-encoders
https://cocodataset.org/#download

```bash
>> cd clip_benchmark
>> python setup.py install

>> clip_benchmark eval --dataset cifar10 cifar100 --task=zeroshot_classification --model source source+aligner target --pretrained NONE  --model_type our_experimental_models  --model_cache_dir=out/clip-to-e5--linear5/  --output="benchmark_{dataset}_{model}_{task}.json.json" --batch_size=64
#^ TODO save to a file in the model-dir
```

# update the submodule (= `pull`)
```bash
git submodule update --recursive --remote
```

# Dataset Downloads

- Flicker: https://github.com/awsaf49/flickr-dataset