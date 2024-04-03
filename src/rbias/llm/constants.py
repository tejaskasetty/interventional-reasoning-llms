
HF_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
{{You are a helpful assistant for causal reasoning.}}<<SYS>>
###

Previous Conversation:
'''
{history}
'''

{{{input}}}[/INST]

"""

LLAMA_PROMPT_TEMPLATE = HF_PROMPT_TEMPLATE

MISTRAL_PROMPT_TEMPLATE = HF_PROMPT_TEMPLATE

STARCHAT_PROMPT_TEMPLATE = HF_PROMPT_TEMPLATE

SYS_MSG = "You are a helpful assistant for causal reasoning."

RESPONSE_FORMAT = "\nPlease provide the answer in <answer></answer> tag. The only accepted values are 'yes' or 'no'."