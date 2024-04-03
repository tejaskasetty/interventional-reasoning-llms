from .gpt import GPT
from .hf import Llama, Mistral, Starchat
from .model import Model

def init_llm(model: Model):
    return LLMs[model](model)

LLMs = {
    Model.GPT_35: GPT,
    Model.GPT_4: GPT,
    Model.GPT_4_TURBO: GPT,
    Model.LLAMA: Llama,
    Model.MISTRAL: Mistral,
    Model.STARCHAT: Starchat

}
