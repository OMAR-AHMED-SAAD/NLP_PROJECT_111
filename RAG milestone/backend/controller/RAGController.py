from .BaseController import BaseController
from components import RAG, ChatModelFactory
from typing import List
from enums import ChatModelTypes
from db import VectorDB
from prompts import *
from logger import get_logger
from enums import PromptEnums

logger = get_logger(__name__)

class RAGController(BaseController):
    '''
    The RAGController class is a subclass of the BaseController class. It initializes the RAG model with an embeddings model name, a vector database, a chat model factory, a system prompt, a k value, and a memory size.
    '''

    def __init__(self,
                 embeddings_model_name: str,
                 vdb: VectorDB,
                 chat_factory: ChatModelFactory,
                 system_prompt: str = "system_prompt",
                 k: int = 1,
                 memory_size: int = 10,
                 ingest: bool = True):
        '''
        Initializes the RAGController with an embeddings model name, a vector database, a chat model factory, a system prompt, a k value, and a memory size.

        Args:
        - embeddings_model_name (str): The embeddings model name
        - vdb (VectorDB): The vector database
        - chat_factory (ChatModelFactory): The chat model factory
        - system_prompt (str): The system prompt
        - k (int): The k value
        - memory_size (int): The memory

        '''
        super().__init__(embeddings_model_name=embeddings_model_name,
                         vdb=vdb,
                         chat_factory=chat_factory,
                         system_prompt=system_prompt,
                         k=k)
        self.memory_size = memory_size
        self.ingest = ingest

    def resolve_prompt(self) -> str:
        '''
        Resolves the system prompt based on the provided system prompt type.

        Returns:
        - str: The resolved system prompt
        '''

        if self.system_prompt == PromptEnums.SYSTEM_PROMPT.value:
            return SYSTEM_PROMPT
        elif self.system_prompt == PromptEnums.COT_PROMPT.value:
            return COT_PROMPT
        elif self.system_prompt == PromptEnums.ZERO_SHOT_PROMPT.value:
            return ZERO_SHOT_PROMPT


    def initialize(self, texts: List[str] = None) -> RAG:
        '''
        Initializes the RAG model.

        Returns:
        - RAG: The initialized RAG model
        '''
        if self.ingest:
            self.vdb.ingest(texts=texts)
            logger.info(f"RAGController: Ingested {len(texts)} texts into the vector database.")
        
        chat_llm = self.chat_factory.create_model(
            model_class=self.chat_factory.config.MODEL_CHAT_CLASS,
            model_type=ChatModelTypes.CHAT.value)
        
        refinement_llm = self.chat_factory.create_model(
            model_class=self.chat_factory.config.MODEL_REFINEMENT_CLASS,
            model_type=ChatModelTypes.REFINEMENT.value)
        
        rag = RAG(rag_llm=chat_llm,
                  refinement_llm=refinement_llm,
                  vdb=self.vdb,
                  texts=None,
                  system_prompt=self.resolve_prompt(),
                  k=self.k,
                  memory_size=self.memory_size)
        return rag