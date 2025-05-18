from enum import Enum

class PromptEnums(Enum):
    '''
    An enumeration of the prompt services.
    '''
    COT_PROMPT = "cot_prompt"
    SYSTEM_PROMPT = "system_prompt"
    ZERO_SHOT_PROMPT = "zero_shot_prompt"