#  https://github.com/Lynn1 update 2024.4.29
#  llama3_70Bf model is originally divided into 8 shards, which only supports 8 GPUs environment.
#  This `concat_llama3_70Bf.py` script is used to concatenate the llama3_70Bf, to support 2 or 4 GPUs environments.

import json
import os
import fire
import torch
from tqdm import tqdm
from typing import List, Union,Tuple

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def concat_llama3_70Bf(
    input_base_path:str,
    output_base_path:str,
    num_original_shards=8,
    num_shards=4,
):
    assert num_original_shards % num_shards == int(0) , f"num_original_shards % num_shards !=0."

    # load_llama3 ckp
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location=f'cpu') #cuda:{i // 2}
            for i in range(num_original_shards)
    ]
    
    # load params
    params = read_json(os.path.join(input_base_path, "params.json"))
    # num_shards = num_shards
    # params = params.get("model", params)      #no "model" in llama3 params
    n_layers = params["n_layers"]               #80
    n_heads = params["n_heads"]                 #64
    n_heads_per_shard = n_heads // num_shards   #64/4=16
    dim = params["dim"]                         #8192
    dims_per_head = dim // n_heads              #8192/64=128
    base = params.get("rope_theta", 10000.0)    #500000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    max_position_embeddings = 8192
    vocab_size = params["vocab_size"]           #128256
    num_key_value_heads = params["n_kv_heads"]  #8 for GQA / MQA The number of key-value headers for the attention mechanism
    num_local_key_value_heads = n_heads_per_shard // num_key_value_heads #16/8=2
    key_value_dim = dim // num_key_value_heads  #8192/8=1024

    # permute for sliced rotary
    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    
    # Rsplit Sharded model 
    state_dict = [{} for _ in range(num_shards)] #create num_shards dicts

    def insert(name: str, tensor: Union[List, torch.Tensor]):
        for i in range(num_shards):
            state_dict[i][name] = (
                tensor[i].clone() if isinstance(tensor, list) else tensor
            )

    def insert_chunk(name: str, tensor: torch.Tensor, dim: int):
        tensors = tensor.chunk(num_shards, dim=dim)
        # Divide evenly according to the number of target blocks
        for i, tensor in enumerate(tensors):
            state_dict[i][name] = tensor.clone()

    def insert_cat(name: str, tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], concat_dim: int):
        # Splice the original Chunk based on the number of target blocks
        new_tensors = []
        for j in range(num_shards):
            k = j * (num_original_shards // num_shards)
            a = (num_original_shards // num_shards)
            new_tensors.append(torch.cat(tensors[k:k+a], dim=concat_dim)) 
        for i, tensor in enumerate(new_tensors):
            state_dict[i][name] = tensor.clone()

    # 1. tok_embeddings
    concat_dim = 0 #if llama_version == 3 else 1
    insert_cat("tok_embeddings.weight", [loaded[i]["tok_embeddings.weight"] for i in range(num_original_shards)], concat_dim)

    # 2. hidden layers
    # Note that:
    #       attention.w{q,k,v,o}, 
    #       feed_fordward.w[1,2,3], 
    #       attention_norm.weight,  
    #       ffn_norm.weight 
    # share the same storage object, saving attention_norm and ffn_norm will save other weights too, 
    # which is redundant as other weights will be stitched from multiple shards. 
    # To avoid that, they are cloned.
    for layer_i in range(n_layers):
        # aq ak 
        insert_cat(f"layers.{layer_i}.attention.wq.weight", [loaded[i][f"layers.{layer_i}.attention.wq.weight"] for i in range(num_original_shards)], 0)
        insert_cat(f"layers.{layer_i}.attention.wk.weight", [loaded[i][f"layers.{layer_i}.attention.wk.weight"] for i in range(num_original_shards)], 0)

        # av ao f1 f2 f3
        insert_cat(f"layers.{layer_i}.attention.wv.weight", [loaded[i][f"layers.{layer_i}.attention.wv.weight"] for i in range(num_original_shards)], 0)
        insert_cat(f"layers.{layer_i}.attention.wo.weight", [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_original_shards)], 1)
        insert_cat(f"layers.{layer_i}.feed_forward.w1.weight", [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_original_shards)], 0)
        insert_cat(f"layers.{layer_i}.feed_forward.w3.weight", [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_original_shards)], 0)
        insert_cat(f"layers.{layer_i}.feed_forward.w2.weight", [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_original_shards)], 1)
        
        # anorm fnorm
        insert(f"layers.{layer_i}.attention_norm.weight", loaded[0][f"layers.{layer_i}.attention_norm.weight"])
        insert(f"layers.{layer_i}.ffn_norm.weight", loaded[0][f"layers.{layer_i}.ffn_norm.weight"])

    # 3. norm.weight
    insert("norm.weight", loaded[0][f"norm.weight"])

    # 4. output.weight
    insert_cat("output.weight", [loaded[i][f"output.weight"] for i in range(num_original_shards)], 0)

    for i in tqdm(range(num_shards), desc="Saving checkpoint shards"):
        torch.save(
            state_dict[i], os.path.join(output_base_path, f"consolidated.{i:02d}.pth")
        )
    
    return

def main(
        input_base_path:str="Meta-Llama-3-70B-Instruct/original",
        output_base_path:str="Meta-Llama-3-70B-Instruct-2shards",
        num_original_shards: int=8,
        num_new_shards: int=4,
):
    os.makedirs(output_base_path, exist_ok=True)
    concat_llama3_70Bf(input_base_path,output_base_path,num_original_shards,num_new_shards)

if __name__ == "__main__":
    fire.Fire(main)
