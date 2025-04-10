from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from pydantic import BaseModel
from collections import defaultdict
from typing import Callable

class TestCase(BaseModel):
    object: str
    answer: str

color_prompts = [
    lambda obj: f"In a single word, the color that most associated with {obj}s is",
    lambda obj: f"The color of {obj} is usually",
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
    TestCase(object="nature", answer="purple"),
    TestCase(object="denim", answer="blue"),
    TestCase(object="dandelion", answer="yellow"),
    TestCase(object="fog", answer="gray"),
    TestCase(object="frog", answer="green"),
    TestCase(object="sunflower", answer="yellow"),
    TestCase(object="brick", answer="red"),
    TestCase(object="wheat", answer="yellow"),
    TestCase(object="tangerine", answer="orange"),
]


def evaluate(prompts: list[Callable[[str], str]], cases: list[TestCase], attempts: int = 50):
    checkpoint = "HuggingFaceTB/SmolLM2-360M"
    device = "cuda" # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    
    results = {}
    per_object_stats = defaultdict(int)  # Track per-object stats
    total_attempts_per_object = len(prompts) * attempts  # Total attempts for each object
    best = 0
    best_text = ''

    for prompt in prompts:
        results[prompt("<OBJ>")] = {}
        prompt_res: dict = results[prompt("<OBJ>")]
        
        for test in cases:
            input_ids = tokenizer.encode(prompt(test.object), return_tensors="pt").to(device)
            success = 0
            
            for i in range(attempts):
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
                
                if test.answer.lower() in new_text.lower():
                    success += 1
                    per_object_stats[test.object] += 1  # Increment total success for this object
                
            prompt_res[test.object] = success
            
        avg = round(sum(prompt_res.values()) / len(prompt_res.values()), 2)
        prompt_res["avg"] = avg
        
        if avg > best:
            best = avg
            best_text = prompt("<OBJ>")

    # Calculate per-object success rates
    per_object_success_rates = {obj: round(successes / total_attempts_per_object, 2) 
                              for obj, successes in per_object_stats.items()}

    print(f"Repeats per object = {attempts}")
    print(json.dumps(results, indent=2))
    print(f"\nBest performer: {best_text}")
    print("\nPer-object stats across all prompts:")
    sorted_stats = dict(sorted(per_object_success_rates.items(), key=lambda x: x[1], reverse=True))
    print(json.dumps(sorted_stats, indent=2))
    
    # Return the complete results dictionary along with other useful statistics
    return {
        "prompt_results": results,
        "best_prompt": best_text,
        "best_score": best,
        "per_object_success_rates": sorted_stats
    }

if __name__ == "__main__":
    evaluate(color_prompts, color_cases, attempts=50)