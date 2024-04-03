import re

from .model import Model

class BaseLLM:
    
    def __init__(self, model_name: str, model_id):
        self.model_name = model_name
        self.model_id = model_id
    
    def __call__(self, input):
        raise NotImplementedError()
    
    def chat_and_retry(self):
        raise NotImplementedError()
    
    @property
    def cls(self):
        return type(self).__name__


def extract_html_tags(text, keys):
        content_dict = {}
        keys = set(keys)
        for key in keys:
            pattern = f"<{key}>(.*?)</{key}>"
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                content_dict[key] = [match.strip() for match in matches]
        return content_dict

def find_answer(text):
    pattern = r"\b(?:yes|no)\b"
    matches = re.findall(pattern, text)
    return matches
    

def parse_response(input_string, parse_failed = False):
    try:
        answer = extract_html_tags(input_string.lower(), ["answer"])["answer"][0]
        if answer not in ['yes', 'no'] :
            return (
                answer,
                False,
                "Error: An <answer></answer> tag was found, but the contained value is invalid. The only accepted values are 'yes' or 'no'.",
            )
    except Exception as err:
        matches = find_answer(input_string.lower())
        if parse_failed and matches:
            return (matches[0], True, None)
            
        return (
            input_string,
            False,
            f"Error: Failed to extract the answer - missing <answer></answer> tag. Please provide the answer in <answer></answer> tag. The only accepted values are 'yes' or 'no'.",
        )
    return (answer, True, None)