
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from functions import ModelWrapper
if __name__ == "__main__":
    test_models = ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-1.5B", "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2.5-3B", "meta-llama/Llama-3.2-3B"]
    # "google/gemma-3-1b-pt"
    # Gemma is weird and doesn't look the same as any of the above models

    for model in test_models:
        prompt = "n^2 up to n=10: [1, 4, 9,"
        answer = "4, 9, 16"
        modelw = ModelWrapper(model=model)
        max_new = 15

        print('\n' * 5)
        print(f"Model: {model}")
        try:
            out_unmod = modelw.generate(prompts=prompt, max_new=max_new)
            out_unmod = ''.join(list(out_unmod.values())[0])
            print(f"Unmod: {out_unmod}")

            #for top_n in range(5, int(modelw.hdim * 0.1), int(modelw.hdim * 0.01)):
            n = 1
            base = 1.2
            while True:
                top_n = int(base ** n)
                n += 1
                idxs = list(set(modelw.get_top_and_mean_acts(texts=[prompt], top_n=top_n)))

                def ablate(module, input, output):
                    # print_shape(input)
                    # print_shape(output)
                    # print(idxs)
                    output[0][0, -1, list(idxs)] *= 0.1
                    return output

                out_mod = modelw.generate(prompts=prompt, hook=ablate, hook_layers=list(range(modelw.layers)), max_new=max_new)

                out_mod = ''.join(list(out_mod.values())[0])
                
                # print(f"Ablating top {len(idxs)} ({len(idxs)/modelw.hdim*100:.1f} %)")
                # print("=" * 50)
                # print(f"{out_mod}")
                # print("=" * 50)
                
                if answer not in out_mod:
                    print(f"{model} failed after ablating {len(idxs)} ({len(idxs)/modelw.hdim*100:.1f} %)")
                    print("=" * 30)
                    print('\n')
                    del modelw # required so vram is freed
                    break
            
        except KeyboardInterrupt:
            # skip to next model
            continue
        except Exception as e:
            raise e