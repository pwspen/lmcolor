from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaMLP
import torch
import json
from pydantic import BaseModel
from collections import defaultdict
from typing import Callable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

 

def gen_token_with_hiddens(model: str, cases: list[str], device: str = "cuda", verbose=False) -> tuple[list[torch.Tensor], str]: # [hiddens, new_token]
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model).to(device)
    # assert isinstance(model, LlamaForCausalLM), f"Model must be a LlamaForCausalLM instance instead of {type(model)}"
    # layer: LlamaDecoderLayer = model.model.layers[0]
    # mlp: LlamaMLP = layer.mlp
    target_model_type = LlamaForCausalLM
    
    hiddens = {}
    
    for text in cases:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        if verbose:
            print(f"Input IDs for {model=}, {text=}: {input_ids}")
        
        hiddens[text] = []

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )
            # outputs.hidden_states shape: [batch_size=1, layers, 1, sequence_length, hidden_size]
            try:
                all_layers_hidden = [layer_output[0, -1, :] for layer_output in outputs.hidden_states[0]]
            except AttributeError:
                raise ValueError(f"Model hidden state not in expected format (expecting {target_model_type} format)")

            hiddens[text].append(all_layers_hidden)
            
            tok = outputs.sequences[0, -1].unsqueeze(0)
            tok = tokenizer.decode(tok, skip_special_tokens=True)

    return (hiddens, tok)

def compute_group_similarities(hiddens: dict[str, list[torch.Tensor]]):
    # hiddens: {name: [[layer1, ...], [[layer1, ...], ...]}

    # if name is just prompt, len(hiddens[name]) = 1
    # and diagonal of sim matrix will be 1

    # if name is object, len(hiddens[name]) = n_objects_in_group
    # and diagonal will not be 1 (represents average inter-group similarity)
    n_groups = len(hiddens)
    group_names = list(hiddens.keys())
    
    # Determine number of layers from first entry
    first_group = list(hiddens.values())[0]
    first_entry = first_group[0]
    n_layers = len(first_entry)
    
    # Initialize 3D similarity matrix [layers, n_groups, n_groups]
    similarity_matrix = torch.zeros(n_layers, n_groups, n_groups)
    
    # Compute similarities between all pairs of groups for each layer
    for layer in range(n_layers):
        for i, name_i in enumerate(group_names):
            for j, name_j in enumerate(group_names):
                if i <= j:  # only upper triangle
                    # Extract specified layer
                    tensors_i = torch.stack([entry[layer] for entry in hiddens[name_i]])
                    tensors_j = torch.stack([entry[layer] for entry in hiddens[name_j]])
                    
                    a = tensors_i.unsqueeze(1)  # Shape: [len(group_i), 1, embedding_dim]
                    b = tensors_j.unsqueeze(0)  # Shape: [1, len(group_j), embedding_dim]
                    
                    # Compute cosine similarity between all pairs
                    sim = F.cosine_similarity(a, b, dim=2).mean().item()
                    
                    similarity_matrix[layer, i, j] = sim
                
    return similarity_matrix, group_names

if __name__ == "__main__":
    test_models = ["Qwen/Qwen2-1.5B", "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "google/gemma-3-1b-pt"]

    prompt = lambda obj: f"The color most associated with {obj}, in a single word:"

    cases = [
        prompt("gold"),
        prompt("banana"),
    ]

    for model in test_models:
        hiddens, tok = gen_token_with_hiddens(model=model, cases=cases)
        sim_matrix, names = compute_group_similarities(hiddens)
        sim_mat_dim = len(cases)
        avgs = []
        for layer in range(len(sim_matrix)):
            # This takes the upper triangle of the similarity matrix and takes mean
            # Representing the average similarity across all prompt pairs
            # So the number is most meaningful for cases = 2
            # (because it's a single number instead of average of 3 or more similarities)
            # (2 cases -> 1 similarity, 3 cases => 3 similarities)
            # 
            triu_indices = torch.triu_indices(sim_mat_dim, sim_mat_dim, offset=1)

            # len = (cases^2 - cases) / 2
            sims = sim_matrix[layer][triu_indices[0], triu_indices[1]]
            
            self_avg = sims.mean().item()
            avgs.append(self_avg)
        plt.scatter(range(len(avgs)), avgs, label=model.split("/")[-1])
        plt.plot(range(len(avgs)), avgs, '-', alpha=0.7)
    plt.title('\n'.join([f"P{n}: {case}" for n, case in enumerate(cases)]))
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("Average Cosine Similarity")
    plt.show()