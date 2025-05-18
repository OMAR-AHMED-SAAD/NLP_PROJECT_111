from enum import Enum

class ChatEnums(Enum):
    '''
    An enumeration of the chat services.
    '''
    OLLAMA = "ollama"
    GOOGLE = "google"
    FINETUNED_ENCODER = "finetuned_encoder"
