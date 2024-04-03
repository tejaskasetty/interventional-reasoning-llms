# Example commands for prompt generation
# python ./scripts/generate_prompts.py --vt rchar -t 5 --vc 3 --dp experiments/prompts  # random char
# python ./scripts/generate_prompts.py --vt cs -t 5 --vc 3 --dp experiments/prompts  # common sensical
# python ./scripts/generate_prompts.py --vt adv -t 5 --vc 3 --dp experiments/prompts  # adversarial

# Example commands for submitting prompts

# python ./scripts/submit_prompts.py -m gpt-4 -p "./experiments/prompts/prompt_rchar_1711052432" -r ./experiments/results/ -i 1 # gpt-4
# python ./scripts/submit_prompts.py -m gpt-4 -p "./experiments/prompts/prompt_cs_1711052444" -r ./experiments/results/ -i 1 # gpt-4
# python ./scripts/submit_prompts.py -m gpt-4 -p "./experiments/prompts/prompt_adv_1711052457" -r ./experiments/results/ -i 1 # gpt-4

# python ./scripts/submit_prompts.py -m gpt-3.5-turbo -p "./experiments/prompts/prompt_rchar_1711052432" -r ./experiments/results/ -i 1 # gpt-3.5-turbo
# python ./scripts/submit_prompts.py -m gpt-3.5-turbo -p "./experiments/prompts/prompt_cs_1711052444" -r ./experiments/results/ -i 1 # gpt-3.5-turbo
# python ./scripts/submit_prompts.py -m gpt-3.5-turbo -p "./experiments/prompts/prompt_adv_1711052457" -r ./experiments/results/ -i 1 # gpt-3.5-turbo

# python ./scripts/submit_prompts.py -m gpt-4-turbo -p "./experiments/prompts/prompt_rchar_1711052432" -r ./experiments/results/ -i 1 # gpt-4-turbo
# python ./scripts/submit_prompts.py -m gpt-4-turbo -p "./experiments/prompts/prompt_cs_1711052444" -r ./experiments/results/ -i 1 # gpt-4-turbo
# python ./scripts/submit_prompts.py -m gpt-4-turbo -p "./experiments/prompts/prompt_adv_1711052457" -r ./experiments/results/ -i 1 # gpt-4-turbo

# python ./scripts/submit_prompts.py -m llama-2-7b -p "./experiments/prompts/prompt_rchar_1711052432" -r ./experiments/results/ -i 1 # llama-2-7b
# python ./scripts/submit_prompts.py -m llama-2-7b -p "./experiments/prompts/prompt_cs_1711052444" -r ./experiments/results/ -i 1 # llama-2-7b
# python ./scripts/submit_prompts.py -m llama-2-7b -p "./experiments/prompts/prompt_adv_1711052457" -r ./experiments/results/ -i 1 # llama-2-7b

# python ./scripts/submit_prompts.py -m gpt-3.5-turbo -p "./experiments/prompts/prompt_rchar_1711754018" -r ./experiments/results/ -i 2 # gpt-3.5-turbo
# python ./scripts/submit_prompts.py -m gpt-4 -p "./experiments/prompts/prompt_rchar_1711754018" -r ./experiments/results/ -i 2 # gpt-4
# python ./scripts/submit_prompts.py -m gpt-4-turbo -p "./experiments/prompts/prompt_rchar_1711754018" -r ./experiments/results/ -i 2 # gpt-4-turbo

# python ./scripts/submit_prompts.py -m gpt-3.5-turbo -p "./experiments/prompts/prompt_cs_1711754038" -r ./experiments/results/ -i 1 # gpt-3.5-turbo
# python ./scripts/submit_prompts.py -m gpt-4 -p "./experiments/prompts/prompt_cs_1711754038" -r ./experiments/results/ -i 1 # gpt-4
# python ./scripts/submit_prompts.py -m gpt-4-turbo -p "./experiments/prompts/prompt_cs_1711754038" -r ./experiments/results/ -i 1 # gpt-4-turbo

python ./scripts/submit_prompts.py -m gpt-3.5-turbo -p "./experiments/prompts/prompt_adv_1711754160" -r ./experiments/results/ -i 1 # gpt-3.5-turbo
python ./scripts/submit_prompts.py -m gpt-4 -p "./experiments/prompts/prompt_adv_1711754160" -r ./experiments/results/ -i 1 # gpt-4
python ./scripts/submit_prompts.py -m gpt-4-turbo -p "./experiments/prompts/prompt_adv_1711754160" -r ./experiments/results/ -i 1 # gpt-4-turbo
