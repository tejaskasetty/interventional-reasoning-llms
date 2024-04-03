from tqdm import tqdm 
from warnings import warn
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from .base import BaseLLM, parse_response
from .constants import RESPONSE_FORMAT, SYS_MSG
from .model import Model

class GPT(BaseLLM):
    def __init__(self, model: Model = Model.GPT_4, temparature = 0, n = 1):
        super().__init__(model.values[0], model.values[1])
        self.llm = ChatOpenAI(model_name=self.model_id, 
                            temperature=temparature, n=n)

    def __call__(self, prompts):
        return self.__chat_and_retry(prompts, n_retry=10)

    def __chat_and_retry(self, prompts, n_retry = 10):
        inputs = list((prompts + RESPONSE_FORMAT).values)
        results = []
        success_no_retry, success_retry, failure = 0, 0, 0
        for i, input in tqdm(enumerate(inputs), total = len(inputs),
                      desc = "Prompts: "):
            messages  = []
            messages.append(SystemMessage(content = SYS_MSG))
            messages.append(HumanMessage(content = input))
            for j in range(n_retry + 1):
                answer = self.llm(messages)
                response = answer.content
                value, valid, retry_message = parse_response(response.lower(), j != 0)
                if valid:
                    success_no_retry += int(j == 0)
                    success_retry += int(j != 0)
                    break
                msg = f"Query {i} failed. Retrying {j+1}/{n_retry}.\n[LLM]:\n{response}\n[User]:\n{retry_message}"
                warn(msg, RuntimeWarning)
                input = retry_message
            
            if not valid:
                failure += 1
                value = "err"
                msg = f"Error - query {i} failed. Could not parse response after {n_retry} retries."
                warn(msg, RuntimeWarning)
            results.append((value, response))
        query_log = (success_no_retry, success_retry, failure)
        return results, query_log
