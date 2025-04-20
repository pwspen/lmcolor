
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from functions import plot_similarity_traces
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from functions import gen_token_with_hiddens




def gen_stream(model: str, text: str | list[str], device: str = "cuda", new: int = 1) -> list[str]:
    print("init tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model)
    print("init model")
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model).to(device)
    # model.generation_config.pad_token_id = tokenizer.pad_token_id
    if isinstance(text, str):
        print("str detected")
        texts = [text]
    elif isinstance(text, list):
        print("list detected")
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
        # outputs.hidden_states shape: [batch_size=1, layers, 1, sequence_length, hidden_size]
        all_layers_hidden = [layer_output[0, -1, :] for layer_output in outputs.hidden_states[0]]
        out_seq = outputs.sequences[0]
        out_seq = tokenizer.decode(out_seq, skip_special_tokens=True)
        outs.append(out_seq)
    return outs



if __name__ == "__main__":
    test_models = ["HuggingFaceTB/SmolLM2-360M", "meta-llama/Llama-3.2-1B", "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2-1.5B", "meta-llama/Llama-3.2-3B"]
    # "google/gemma-3-1b-pt"
    # Gemma is weird and doesn't look the same as any of the above models

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

    for model in test_models:
        try:
            # fruits = {
            #     "Citrus": ["lemon", "lime", "orange", "grapefruit", "tangerine", "pomelo", "mandarin"],
            #     "Berries": ["strawberry", "blueberry", "raspberry", "blackberry", "cranberry"],
            #     "Tropicals": ["banana", "pineapple", "mango", "papaya", "kiwi", "guava"],
            #     "Stone fruits": ["peach", "plum", "cherry", "apricot", "nectarine"]
            # }

            # template: str = lambda name, first: f"{name}: [{first},"
            # texts = [template(name, item) for name, items in fruits.items() for item in items]

            # print(texts)

            outs = gen_stream(model=model, text="hard times create strong", new=20)
            print(outs)
            for t in outs:
                print(t)
                print("=" * 50)
            exit()
            # plot_similarity_traces(
            #     model=model,
            #     pairs=pairs,
            #     template=template,
            # )
        except KeyboardInterrupt:
            # skip to next model
            continue