from enum import Enum

class Model(Enum):
    GPT_35 = 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'
    GPT_4 = 'gpt-4', 'gpt-4'
    GPT_4_TURBO = 'gpt-4-turbo', 'gpt-4-turbo-preview'
    LLAMA = 'llama-2-7b', 'meta-llama/Llama-2-7b-chat-hf'
    MISTRAL = 'mistral-7b', 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    STARCHAT = 'starchat-beta', 'HuggingFaceH4/starchat-beta'

    def __new__(cls, *values):
        obj = object.__new__(cls)
        # first value is canonical value
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj

    def __repr__(self):
        return '<%s.%s: %s>' % (
                self.__class__.__name__,
                self._name_,
                ', '.join([repr(v) for v in self._all_values]),
                )

    @property
    def values(self):
        return self._all_values