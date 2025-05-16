from prompts import SYSTEM_PROMPT
from abc import ABC, abstractmethod
from typing import List
from db import VectorDB
from components import ChatModelFactory
class BaseController(ABC):
    def __init__(self, 
                 embeddings_model_name: str,
                 vdb: VectorDB,
                 chat_factory: ChatModelFactory,
                 system_prompt: str=SYSTEM_PROMPT,
                 k: int= 1):
        '''
        Initializes the BaseController with an embeddings model name, a vector database, a chat model factory, a system prompt, and a k value.

        Args:
        - embeddings_model_name (str): The embeddings model name
        - vdb (VectorDB): The vector database
        - chat_factory (ChatModelFactory): The chat model factory
        - system_prompt (str): The system prompt
        - k (int): The k value
        '''
        self.embeddings_model_name = embeddings_model_name
        self.system_prompt = system_prompt
        self.k = k
        self.vdb = vdb
        self.chat_factory = chat_factory
  

    @abstractmethod
    def initialize(self, texts: List[str]):
        pass