Script for comparing similarity of LLM hidden states between different prompts, across all layers.

Built on Transformers library, works with any model that has the same hidden state structure as LlamaForCausalLM (most models). Only intended for pretrained (not instruct) models.

Run with `uv run prompt_hidden_similarity.py --prompts "prompt1" "prompt2"`

Example comparing hidden states for prompts P0 and P1:

![sim](examples/sim.png)

Less similar prompts show less structured similarity:

![somesim](examples/somesim.png)

Note magnitude: the least similar that any model+layer gets on the same-structure prompts is about the most similar that any model+layer gets on the different-structure prompts.

What do all the squiggles mean? Why do most models have a "bounce" for the same-structure prompts? Why does the last layer increase similarity in one case but decreases it in the other? Who knows :)

Unfortunately the prompt structure seems to matter a lot more than the concept encoded in the next token:

![nosim](examples/nosim.png)

So this fails at my original goal but it does surface some interesting patterns. And in some cases changing one word does significantly change things:

![oneword](examples/oneword.png)