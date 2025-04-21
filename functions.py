import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Callable
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch
import torch.nn.functional as F
from collections import namedtuple

# This is to make it easy to understand nested structures especially
# when they involve lists and tuples as well as tensors
def print_shape(tensor: list | tuple | torch.Tensor, level: int = 0):
    ty = type(tensor)
    if ty == torch.Tensor:
        print(f"Dim{level}: {ty}, shape: {tensor.shape}")
    elif ty in [tuple, list]:
        print(f"Dim{level}: {ty}, len: {len(tensor)}")
    else:
        print(f"Dim{level}: {ty}")
    try:
        next = tensor[0]
        print_shape(next, level + 1)
    except (ValueError, IndexError):
        print("No size")
        print(tensor)

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

class ModelWrapper:
    def __init__(self, model: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model).to(device)
        self.device = device
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.layers = self.model.config.num_hidden_layers
        self.hdim = self.model.config.hidden_size

        # assert isinstance(model, LlamaForCausalLM), f"Model must be a LlamaForCausalLM instance instead of {type(model)}"
        # layer: LlamaDecoderLayer = model.model.layers[0]
        # mlp: LlamaMLP = layer.mlp
        # target_model_type = LlamaForCausalLM

    def get_top_and_mean_acts(self, texts, top_n, min_same_max_act_frac=0.5):
        hiddens, toks = self.generate(prompts=texts, max_new=1, return_hiddens=True)

        # Tensor[cases, layers, hdim]
        mat = torch.stack(list(hiddens.values()))

        sorted, sortedidx = torch.sort(torch.abs(mat), dim=-1, descending=True)

        # print(sortedidx[0, :, :5])

        val, count = torch.mode(sortedidx[0, :, :top_n], dim=0)

        # print(f"Top activation idx: {val}, count: {count}")

        layers = self.model.config.num_hidden_layers

        if any(count / layers < min_same_max_act_frac):
            text = f"Less than {min_same_max_act_frac*100:.0f} % of layers have same largest activation"
            # raise ValueError(text)
            # print(text)    

        # print(val)
        return val.tolist()

        sorted = sorted[:, :, :5] # Top 5 activations by layer

        sorted_means = torch.mean(sorted, dim=0)

        means = torch.mean(mat, dim=[0, -1]) # Mean activation by layer, averaging cases and hdim

        # for layer in range(sorted_means.shape[0]):
        #     tops = "[" + ', '.join([f"{val:.3g}x" for val in (sorted_means[layer, :] / means[layer]).tolist()]) + "]"
        #     print(f"L{layer}: top: {tops}, topidx: {sortedidx[0, layer, 0]}") # idx goes from first case only cause can't be averaged

    def register_hook(self, hook, layers):
        handles = []
        for layer in layers:
            # Register the hook for each layer
            if hasattr(self.model, 'model'):
                # For models with a 'model' attribute (like LlamaForCausalLM)
                layer_module = self.model.model.layers[layer]
            else:
                # For models without a 'model' attribute
                layer_module = self.model.layers[layer]

            if hasattr(layer_module, 'register_forward_hook'):
                handle = layer_module.register_forward_hook(hook)
                handles.append(handle)
            else:
                raise ValueError(f"Layer {layer} does not support forward hooks.")
        
        def remove_hooks():
            for handle in handles:
                handle.remove()

        return remove_hooks

    def generate(self, prompts: dict[str, str] | list[str] | str, return_hiddens: bool = False, hook: Callable | None = None, hook_layers: list[int] | None = None, max_new: int = 50, verbose=False) -> tuple[dict[str, list[torch.Tensor]], dict[str, str]]: # [hiddens, new_token]
        GenerationResponse = namedtuple('GenerationResponse', ['hiddens', 'new_token'])

        if isinstance(prompts, list):
            prompts = {case: case for case in prompts}

        if isinstance(prompts, str):
            prompts = {prompts: prompts}

        hiddens = {}
        toks = {}

        if hook is not None and hook_layers is not None:
            remove_handles = self.register_hook(hook, hook_layers)
        elif hook is not None or hook_layers is not None:
            raise ValueError("Both hook and hook_layers must be provided together")
        else:
            remove_handles = None

        for key, text in prompts.items():
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            if verbose:
                print(f"Input IDs for {self.model=}, {text=}: {input_ids}")

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                # outputs.hidden_states shape: [batch_size=1, layers, 1, sequence_length, hidden_size]
                try:
                    all_layer_outputs = outputs.hidden_states[0]

                    hidden_stack = torch.stack([layer_output[:, -1, :] for layer_output in all_layer_outputs]) # only get from last generated token
                    # shape [num_layers, hidden_size]

                    hidden_stack = hidden_stack.squeeze(1)  # Remove batch dimension to get [num_layers, hidden_size]

                    if verbose:
                        print_shape(hidden_stack)
                    
                except AttributeError:
                    raise ValueError(f"Model hidden state not in expected format (expecting LlamaModelForCausalLM format)")

                if key in hiddens:
                    raise ValueError(f"Key {key} already exists in hiddens dictionary")
                hiddens[key] = hidden_stack
                
                tok = outputs.sequences[0, :]
                # print(f"{tok=}")
                tok = self.tokenizer.batch_decode(tok, skip_special_tokens=True)
                toks[key] = tok

        if remove_handles:
            remove_handles()
        if return_hiddens:
            return (hiddens, toks)
        else:
            return toks

    def plot_similarity_traces(self, pairs: dict[str, str], template: Callable):
        cases = {object: template(object, "color") for object, color in pairs.items()}
        hiddens, toks = self.generate(prompts=cases)
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

            plt.title(f"{self.model}, against {obj1} <{toks[obj1]}>")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Cosine Similarity")
            ax.grid(True, linestyle='--', alpha=0.5) # Add a grid
            # plt.legend()
            plt.tight_layout()
            plt.show()
            # plt.savefig(f"{template(obj="OBJ", query="QRY")}_{model.split('/')[-1]}.png")

template = lambda obj, query: f"The {query} most associated with {obj}, in a single word:"
    # template = lambda obj, query: f"In a single word, {obj} is most associated with the {query}:"    

pairs = {
    'emerald': 'green',
    'strawberry': 'red',
    'lemon': 'yellow',
    'ruby': 'red',
    'grass': 'green',
    'carrot': 'orange', 
    'banana': 'yellow',
    'pumpkin': 'orange',
    'cherry': 'red',
    'tomato': 'red',
    'gold': 'yellow',
    'mustard': 'yellow',
    'yolk': 'yellow',
    'blood': 'red',
    'snow': 'white',
    'peace': 'white',
    'rage': 'red',
    'sapphire': 'blue',
    'lime': 'green',
    'ivory': 'white',
    'jade': 'green',
    'rose': 'red',
    'ocean': 'blue',
    'sky': 'blue',
    'nature': 'green',
    'denim': 'blue',
    'dandelion': 'yellow',
    'frog': 'green',
    'sunflower': 'yellow',
    'brick': 'red',
    'wheat': 'yellow',
    'tangerine': 'orange'
}