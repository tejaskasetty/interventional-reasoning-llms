import os
import numpy as np
import torch
from warnings import warn
from transformers import pipeline, AutoTokenizer
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from .model import Model
from .constants import (HF_PROMPT_TEMPLATE, LLAMA_PROMPT_TEMPLATE, 
                    MISTRAL_PROMPT_TEMPLATE, STARCHAT_PROMPT_TEMPLATE,
                    RESPONSE_FORMAT)
from .base import BaseLLM
from .base import parse_response


class HfLLM(BaseLLM):

    def __init__(self, model: Model, temperature = 0, n = 1):
        super().__init__(model.values[0], model.values[1])
        hf_token =  os.getenv("HF_TOKEN")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model_pipeline = pipeline(
            task="text-generation",
            model=self.model_id,  # model_id can be any huggingface model
            tokenizer=self._tokenizer,
            torch_dtype=torch.bfloat16,  # Saves you memory
            device_map="auto",  # Uses the GPU if available
            max_length=2000,
            do_sample=False, # avoid sampling. choose text with highest log prob.
            num_return_sequences=1,
            eos_token_id=self._tokenizer.eos_token_id,
            token=hf_token, # Necessary for gated models like llama2
            temperature=1,
            top_p=1
        )
        
        self._llm = HuggingFacePipeline(pipeline=self._model_pipeline)
        
        self._memory = ConversationBufferWindowMemory(k=10)
        # self._prompt = PromptTemplate(tempalte=HF_PROMPT_TEMPLATE, input_variables = ['input'])
        # self.chain = ConversationChain(llm=self._llm, prompt=self._prompt, memory=self._memory)
    
    def __call__(self, prompts):
        self._memory.clear()
        return self.__chat_and_retry(prompts, n_retry = 10)

    def __chat_and_retry(self, prompts, n_retry = 10):
        inputs = list((prompts + RESPONSE_FORMAT).values)
        results, retry_indices = self.__apply(inputs)
        num_prompts = len(inputs)
        num_retries = len(retry_indices)
        success_no_retry = num_prompts - num_retries
        success_retry = 0
        failure = 0
        for i in retry_indices:
            self._memory.clear()
            input = inputs[i]
            for j in range(n_retry):
                output = self.chain({ "input" : input})
                response = output['response']
                value, valid, retry_message = parse_response(response.lower(), True)
                if valid:
                    break
                msg = f"Query {i} failed. Retrying {j+1}/{n_retry}.\n[LLM]:\n{response}\n[User]:\n{retry_message}"
                warn(msg, RuntimeWarning)
                input = retry_message
            
            if not valid:
                failure += 1
                value = "err"
                msg = f"Error - query {i} failed. Could not parse response after {n_retry} retries."
                warn(msg, RuntimeWarning)
            results[i] = (value, response)
        
        success_retry = num_prompts - (success_no_retry + success_retry)
        submit_log = (success_no_retry, success_retry, failure)
        return results, submit_log

    def __apply(self, prompts):
        inputs = [{ "input" : prompt, 'history' : ''} for prompt in prompts]
        outputs = self.chain.apply(inputs)
        return self.__process_outputs(outputs)
        
    def __process_outputs(self, outputs):
        results = []
        retry_indices = []
        for i, output in enumerate(outputs):
            response = output['response']
            value, valid, _ = parse_response(response.lower())
            if valid:
                results.append((value, response))
            else:
                results.append(None)
                retry_indices.append(i)
        return results, retry_indices


class Llama(HfLLM):
    
    def __init__(self, model: Model = Model.LLAMA, temperature = 0.0, n = 1):
        super().__init__(model, temperature, n)
        self._prompt = PromptTemplate(template=LLAMA_PROMPT_TEMPLATE,
                                    input_variables=['input', 'history'])
        self.chain = ConversationChain(llm=self._llm, prompt=self._prompt, 
                                       memory=self._memory)


class Mistral(HfLLM):
    
    def __init__(self, model: Model = Model.MISTRAL, temperature = 0.0, n = 1):
        super().__init__(model, temperature, n)
        self._prompt = PromptTemplate(template=MISTRAL_PROMPT_TEMPLATE,
                                    input_variables=['input', 'history'])
        self.chain = ConversationChain(llm=self._llm, prompt=self._prompt, 
                                       memory=self._memory)


class Starchat(HfLLM):
    
    def __init__(self, model: Model = Model.STARCHAT, temperature = 0.0, n = 1):
        super().__init__(model, temperature, n)
        self._prompt = PromptTemplate(template=STARCHAT_PROMPT_TEMPLATE,
                                    input_variables=['input', 'history'])
        self.chain = ConversationChain(llm=self._llm, prompt=self._prompt, 
                                       memory=self._memory)