from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
checkpoint = "HuggingFaceTB/SmolLM2-360M"
device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

initial_text = lambda obj: f"{obj}s are known to have the color"

# Encode initial prompt only once

objects = ["Banana", "Eggplant"]
colors = ["yellow", "purple"]

attempts = 100
results = {}
for obj, col in zip(objects, colors):
    input_ids = tokenizer.encode(initial_text(obj), return_tensors="pt").to(device)
    success = 0
    for i in range(100):
        # Generate only the next token
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Get only the new token ID
        new_token = outputs[0][-1].unsqueeze(0)
        new_text = tokenizer.decode(new_token, skip_special_tokens=True)
        if col.lower() in new_text.lower():
            success += 1
        print(new_text)
    results[obj] = success

print(results)