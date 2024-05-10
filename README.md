# llama3-on-2GPUs

A script tool which recut the original llama3_70B_instruct model into 2 or 4 shards, so that one can run the model efficiently on a `2x80GB` or `4x40GB` GPUs environments.

Lynn1 update 2024.04.29 :

llama3_70Bf model is originally divided into 8 shards, which only supports >=8 GPUs environment.

I added the `concat_llama3_70Bf.py` script in this repo which can concatenate the llama3_70Bf into 2 or 4 shards, so that I can run it on `2x80GB` or `4x40GB` GPUs environments.

> **According to my experiments, loading and reasoning with 2 cards takes less time than with 4 cards.**

### Usage Tips:

### Step1: Recut the original llama3_70B_instruct model depending on the number of graphics cards you want to use

```Bash

#----Example:

original_PATH="Meta-Llama-3-70B-Instruct/original"# replace this with your original model path

MODEL_FOLDER_PATH="Meta-Llama-3-70B-Instruct-2shards"# replace this with your output path

num_original_shards=8# the number of shards in the original model

num_new_shards=2# the number of shards you neeed(depends on the number of GPUs you have)

python ./concat_llama3_70Bf.py  \

    --input_base_path ${original_PATH}/  --output_base_path ${MODEL_FOLDER_PATH}/ \

    --num_original_shards ${num_original_shards} --num_new_shards ${num_new_shards}

```

### Step2: Now you can run the model test with MP=2(or 4, depends on your num_new_shards)

run the `example_chat_completion.py` in the official folder: [meta-llama/llama3: The official Meta Llama 3 GitHub site](https://github.com/meta-llama/llama3)

```Bash
torchrun --nproc_per_node 2 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-70B-Instruct-2shards/ \
    --tokenizer_path Meta-Llama-3-70B-Instruct-2shards/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```
