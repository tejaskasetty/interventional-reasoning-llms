# Interventional Reasoning of LLMs
A benchmark to assess and analyze the interventional causal reasoning capability of Large Language Models.


**Example script commands** 
* To generate prompts: \
`python ./scripts/generate_prompts.py --vt cs -t 5 --vc 3 --dp experiments/prompts`
* To query LLMs: \
`python ./scripts/run_gpt.py -m gpt-4-turbo-preview -p "./experiments/prompts/prompt_rchar_1709153316" -r ./experiments/results/ -i 1`
