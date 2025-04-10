from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from pydantic import BaseModel

class TestCase(BaseModel):
    object: str
    color: str

# Load model and tokenizer
checkpoint = "HuggingFaceTB/SmolLM2-360M"
device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

prompts = [
    lambda obj: f"In a single word, the color that most people associate with {obj}s is",
    lambda obj: f"In a single word, the color that most associated with {obj}s is",
    lambda obj: f"In a single word, the color most strongly associated with {obj}s is",
    lambda obj: f"In a single word, the first color that comes to mind when hearing the word '{obj}' is",
    lambda obj: f"The color most strongly associated with {obj}s is",

]
           
cases = [
    TestCase(object="strawberry", color="red"),
    TestCase(object="emerald", color="green"),
    TestCase(object="sky", color="blue"),
    TestCase(object="snow", color="white"),
    TestCase(object="coal", color="black"),
    TestCase(object="orange", color="orange"),
    TestCase(object="amethyst", color="purple"),
    TestCase(object="chocolate", color="brown"),
    TestCase(object="flamingo", color="pink"),
    TestCase(object="silver", color="gray"),
    TestCase(object="lemon", color="yellow"),
    TestCase(object="ruby", color="red"),
    TestCase(object="grass", color="green"),
    TestCase(object="sapphire", color="blue"),
    TestCase(object="milk", color="white"),
    TestCase(object="raven", color="black"),
    TestCase(object="carrot", color="orange"),
    TestCase(object="lavender", color="purple"),
    TestCase(object="coffee", color="brown"),
    TestCase(object="salmon", color="pink")
]

attempts = 100
results = {}
for prompt in prompts:
    results[prompt("OBJ")] = {}
    for test in cases:
        input_ids = tokenizer.encode(prompt(test.object), return_tensors="pt").to(device)
        success = 0
        for i in range(30):
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
            if test.color.lower() in new_text.lower():
                success += 1
            print(new_text)
        results[prompt("OBJ")][test.object] = success

print(f"Repeats per object = {attempts}")
print(json.dumps(results, indent=2))