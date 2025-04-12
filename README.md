Script for comparing similarity of hidden states between different prompts.

Built on Transformers library, works with any model that has the same hidden state structure as LlamaForCausalLM (most models). Only intended for pretrained (not instruct) models.

Example comparing hidden states for prompts "The color most associated with {obj}, in a single word:" for obj = [gold, banana]:

[]