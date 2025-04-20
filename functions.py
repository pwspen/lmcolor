import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Callable
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch
import torch.nn.functional as F

def gen_token_with_hiddens(model: str, cases: dict[str, str] | list[str], device: str = "cuda", verbose=False) -> tuple[dict[str, list[torch.Tensor]], dict[str, str]]: # [hiddens, new_token]
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model).to(device)
    # assert isinstance(model, LlamaForCausalLM), f"Model must be a LlamaForCausalLM instance instead of {type(model)}"
    # layer: LlamaDecoderLayer = model.model.layers[0]
    # mlp: LlamaMLP = layer.mlp
    target_model_type = LlamaForCausalLM
    
    if isinstance(cases, list):
        cases = {case: case for case in cases}

    hiddens = {}
    toks = {}

    for key, text in cases.items():
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        if verbose:
            print(f"Input IDs for {model=}, {text=}: {input_ids}")
        
        hiddens[key] = []

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

            hiddens[key].append(all_layers_hidden)
            
            tok = outputs.sequences[0, -1].unsqueeze(0)
            tok = tokenizer.decode(tok, skip_special_tokens=True)
            toks[key] = tok

    return (hiddens, toks)

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
                    similarity_matrix[layer, j, i] = sim
                
    return similarity_matrix, group_names

def plot_similarity_traces(model: str, pairs: dict[str, str], template: Callable):
    cases = {object: template(object, "color") for object, color in pairs.items()}
    hiddens, toks = gen_token_with_hiddens(model=model, cases=cases)
    sim_matrix, names = compute_group_similarities(hiddens)
    sim_mat_dim = len(cases)

    results = {}
    layer_count = len(sim_matrix)
    object_names = list(cases.keys())

    for i, obj1 in enumerate(object_names):
        results[obj1] = {}
        for j, obj2 in enumerate(object_names):
            results[obj1][obj2] = [round(sim_matrix[layer][i, j].item(), 5) for layer in range(layer_count)]

    for obj1, result in results.items():
        fig, ax = plt.subplots(figsize=(12, 7)) # Adjust figsize as needed
        lines_data = []
        for obj2, sim_values in result.items():
            assert layer_count == len(sim_values), f"Layer count mismatch: {layer_count} != {len(sim_values)}"
            line, = ax.plot(range(layer_count), sim_values, '-o', alpha=0.7, color=pairs[obj2]) # Label with object name and color of object
            label = f"{obj2} <{toks[obj2]}>"
            lines_data.append({
            'final_y': sim_values[-1],
            'label': f"{obj2}",
            'color': pairs[obj2],
            'pred': toks[obj2]
            })
            # texts.append(ax.text(layer_count-1, sim_values[-1], obj2))
            # ax.annotate(obj2, xy=(layer_count-1, sim_values[-1]), xytext=(5, 0), textcoords='offset points', va='center')

        sorted_lines = sorted(lines_data, key=lambda x: x['final_y'], reverse=True)

        x_pos = layer_count + 0.5  # Position to the right of the plot
        y_spacing = 0.05  # Vertical spacing between labels (adjust as needed)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        start_y = ax.get_ylim()[1] - (y_range * 0.05)  # Start near the top
        
        # Adds labels
        for i, line_data in enumerate(sorted_lines):
            def bbox(color: str):
                if color in ["white", "yellow"]:
                    return dict(facecolor='black', alpha=0.5, edgecolor='none')
                else:
                    return None

            y_pos = start_y - (i * y_spacing * y_range)
            color = line_data["color"]
            ax.text(x_pos, y_pos, line_data['label'], color=line_data["color"], va='center', bbox=bbox(color))
            
            predcolor = "notacolor"
            for col in mcolors.CSS4_COLORS.keys():
                if col in line_data['pred']:
                    predcolor = col
                    break
            ax.text(x_pos + 3, y_pos, f"<{line_data['pred']}>", color="green" if predcolor in color else "red", va='center')
        
            y_pos = start_y - (i * y_spacing * y_range)
            ax.plot([layer_count-1, x_pos], [line_data['final_y'], y_pos], linestyle='--', alpha=0.3, color="black")

        ax.set_xlim(0, layer_count + 2.5)

        plt.title(f"{model}, against {obj1} <{toks[obj1]}>")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.grid(True, linestyle='--', alpha=0.5) # Add a grid
        # plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"{template(obj="OBJ", query="QRY")}_{model.split('/')[-1]}.png")

def gen_stream(model: str, text: str | list[str], device: str = "cuda", new: int = 50) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model)
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model).to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, list):
        texts = text

    outs = []
    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        outputs = model.generate(
                    input_ids,
                    max_new_tokens=new,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=9999
                )
        # print(outputs)
        out_seq = outputs.sequences[0]
        out_seq = tokenizer.decode(out_seq, skip_special_tokens=True)
        outs.append(out_seq)
    return outs