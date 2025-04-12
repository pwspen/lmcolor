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


class TestCase(BaseModel):
    object: str
    answer: str

color_prompts = [
    # lambda obj: f"In a single word, the color that most associated with {obj}s is",
    # lambda obj: f"In a single word, the color that is most associated with {obj}s is",
    # lambda obj: f"In a single word, the color that is the most associated with {obj}s is",
    # lambda obj: f"In a single word, the color that is the most associated with {obj} is",
    # lambda obj: f"In a single word, the color that's most associated with {obj}s is",
    # lambda obj: f"In a single word, the color that most associated with {obj} is",
    lambda obj: f"The color most associated with {obj}, in a single word:",
    # lambda obj: f"The color most strongly associated with {obj}, in a single word:",
    # lambda obj: f"The color most associated with {obj},"
]

color_cases = [
    TestCase(object="strawberry", answer="red"),
    TestCase(object="emerald", answer="green"),
    TestCase(object="flamingo", answer="pink"),
    TestCase(object="lemon", answer="yellow"),
    TestCase(object="ruby", answer="red"),
    TestCase(object="grass", answer="green"),
    TestCase(object="raven", answer="black"),
    TestCase(object="carrot", answer="orange"),
    TestCase(object="banana", answer="yellow"),
    TestCase(object="pumpkin", answer="orange"),
    TestCase(object="eggplant", answer="purple"),
    TestCase(object="cherry", answer="red"),
    TestCase(object="tomato", answer="red"),
    TestCase(object="gold", answer="yellow"),
    TestCase(object="mustard", answer="yellow"),
    TestCase(object="yolk", answer="yellow"),
    TestCase(object="blood", answer="red"),
    TestCase(object="snow", answer="white"),
    TestCase(object="chocolate", answer="brown"),
    TestCase(object="peace", answer="white"),
    TestCase(object="rage", answer="red"),
    TestCase(object="sapphire", answer="blue"),
    TestCase(object="lime", answer="green"),
    TestCase(object="ivory", answer="white"),
    TestCase(object="jade", answer="green"),
    TestCase(object="rose", answer="red"),
    TestCase(object="ocean", answer="blue"),
    TestCase(object="sky", answer="blue"),
    TestCase(object="amethyst", answer="purple"),
    TestCase(object="nature", answer="green"),
    TestCase(object="denim", answer="blue"),
    TestCase(object="dandelion", answer="yellow"),
    TestCase(object="fog", answer="gray"),
    TestCase(object="frog", answer="green"),
    TestCase(object="sunflower", answer="yellow"),
    TestCase(object="brick", answer="red"),
    TestCase(object="wheat", answer="yellow"),
    TestCase(object="tangerine", answer="orange"),
]


def evaluate(prompts: list[Callable[[str], str]], cases: list[TestCase], model: str, attempts: int = 1, verbose=False):
    device = "cuda" # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(model)
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model).to(device)
    # layer: LlamaDecoderLayer = model.model.layers[0]
    # mlp: LlamaMLP = layer.mlp
    
    results = {}
    per_object_stats = defaultdict(int)  # Track per-object stats
    total_attempts_per_object = len(prompts) * attempts  # Total attempts for each object
    best = 0
    best_text = ''
    hiddens = {}
    for prompt in prompts:
        results[prompt("<OBJ>")] = {}
        prompt_res: dict = results[prompt("<OBJ>")]
        
        for test in cases:
            input_ids = tokenizer.encode(prompt(test.object), return_tensors="pt").to(device)
            success = 0
            if test.answer not in hiddens.keys():
                hiddens[test.answer] = []

            for i in range(attempts):
                # Generate only the next token
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        output_hidden_states=True,
                        return_dict_in_generate=True
                    )
                    # [batch, layer]
                    # includes all tokens
                    all_layers_hidden = [layer_output[0, -1, :] for layer_output in outputs.hidden_states[0]]

                    hiddens[test.answer].append(all_layers_hidden)

                # Get only the new token ID
                new_token = outputs.sequences[0, -1].unsqueeze(0)
                new_text = tokenizer.decode(new_token, skip_special_tokens=True)
                
                if test.answer.lower() in new_text.lower():
                    success += 1
                    per_object_stats[test.object] += 1  # Increment total success for this object
                
            prompt_res[test.object] = success
            
        avg = round(sum(prompt_res.values()) / len(prompt_res.values()), 2)
        if verbose:
            print(f"Prompt: {prompt('<OBJ>')}, Success rate: {avg}")
        prompt_res["avg"] = avg
        
        if avg > best:
            best = avg
            best_text = prompt("<OBJ>")

    # Calculate per-object success rates
    per_object_success_rates = {obj: round(successes / total_attempts_per_object, 2) 
                              for obj, successes in per_object_stats.items()}

    if verbose:
        print(f"Repeats per object = {attempts}")
        print(json.dumps(results, indent=2))
        print(f"\nBest performer: {best_text}")
        print("\nPer-object stats across all prompts:")
        sorted_stats = dict(sorted(per_object_success_rates.items(), key=lambda x: x[1], reverse=True))
        print(json.dumps(sorted_stats, indent=2))
    
    # Return the complete results dictionary along with other useful statistics
    return hiddens

def compute_group_similarities(groups_dict, verbose=False):
    # Initialize variables
    n_groups = len(groups_dict)
    group_names = list(groups_dict.keys())
    
    # Determine number of layers from first entry
    first_group = list(groups_dict.values())[0]
    first_entry = first_group[0]
    n_layers = len(first_entry)
    print(f"Number of layers: {n_layers}")
    
    # Initialize 3D similarity matrix [layers, n_groups, n_groups]
    similarity_matrix = torch.zeros(n_layers, n_groups, n_groups)
    
    # Compute similarities between all pairs of groups for each layer
    for layer in range(n_layers):
        for i, name_i in enumerate(group_names):
            for j, name_j in enumerate(group_names):
                if i <= j:  # We only need to compute upper triangular (including diagonal)
                    # Extract the specified layer from each entry
                    tensors_i = torch.stack([entry[layer] for entry in groups_dict[name_i]])
                    tensors_j = torch.stack([entry[layer] for entry in groups_dict[name_j]])
                    
                    # Compute all pairwise similarities
                    a = tensors_i.unsqueeze(1)  # Shape: [len(group_i), 1, embedding_dim]
                    b = tensors_j.unsqueeze(0)  # Shape: [1, len(group_j), embedding_dim]
                    
                    # Compute cosine similarity for all pairs
                    sim = F.cosine_similarity(a, b, dim=2).mean().item()
                    
                    # Store in similarity matrix (symmetrically)
                    similarity_matrix[layer, i, j] = sim
                    similarity_matrix[layer, j, i] = sim  # Symmetric matrix

                    if verbose and layer == 0:  # Only print for first layer to avoid excessive output
                        print(f"Layer {layer}, {name_i} vs {name_j}: {sim:.4f}")
                
    return similarity_matrix, group_names

def plot_similarity_matrix(similarity_matrix, group_names):
    # Convert to numpy array if it's a tensor
    if torch.is_tensor(similarity_matrix):
        sim_matrix_np = similarity_matrix.cpu().numpy()
    else:
        sim_matrix_np = similarity_matrix
    
    # Create figure and axes
    plt.figure(figsize=(10, 8))
    
    # Create heatmap using seaborn for better appearance
    ax = sns.heatmap(
        sim_matrix_np,
        annot=True,  # Show values in cells
        fmt=".2f",   # Format as 2 decimal places
        cmap="viridis",  # Color map (you can use "RdBu_r", "YlGnBu", etc.)
        xticklabels=group_names,
        yticklabels=group_names,
        vmin=0,      # Minimum value (cosine similarity range is -1 to 1, but usually positive)
        vmax=1       # Maximum value
    )
    
    # Add labels and title
    plt.title("Cosine Similarity Between Groups", fontsize=16)
    plt.tight_layout()
    
    # Return the figure for further customization if needed
    return plt.gcf()

if __name__ == "__main__":
    models = ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B")
    last = None
    for model in models:
        hs = evaluate(color_prompts, color_cases, model=model, attempts=1)
        sim_matrix, names = compute_group_similarities(hs)
        if last:
            print(last == sim_matrix)
            last = sim_matrix
        print(len(sim_matrix))
        avgs = []
        for i in range(len(sim_matrix)):
            # print(sim_matrix[i])
            self_avg = sim_matrix[i].diagonal().mean().item()
            # print(f"Layer {i} self avg: {self_avg:.2f}")
            avgs.append(self_avg)
        plt.scatter(range(len(avgs)), avgs)
        plt.plot(range(len(avgs)), avgs, '-', alpha=0.7)
        plt.xlabel("Layer")
        plt.ylabel("Average In-Group Cosine Similarity")
        plt.show()