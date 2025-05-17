from components.LLM import LLM
from db import VectorDB
from typing import List
from langchain.prompts import ChatPromptTemplate
from prompts import SYSTEM_PROMPT
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from collections import deque
from typing import Generator
from logger import get_logger
from ContextLogger import ContextLogger

logger = get_logger(__name__)

class RAG:
    '''
    RAG is a class that represents a Retrieval-Augmented Generation model.
    '''
    def __init__(self,
                 rag_llm: LLM,
                 vdb: VectorDB,
                 texts: List[str],
                 system_prompt: str=SYSTEM_PROMPT,
                 k: int = 1,
                 memory_size: int = 10):
        '''
        Initializes the RAG with a system prompt, a retrieval model, an extractor model, a vector database, and a list of texts.

        Args:
        - rag_llm (LLM): The language model used for generation
        - vdb (VectorDB): The vector database for document retrieval
        - texts (list[str]): A list of texts
        - system_prompt (str): The system prompt
        - k (int): The number of documents to retrieve
        - memory_size (int): The size of the memory
        '''
        self.rag_llm = rag_llm
        self.vdb = vdb
        self.texts = texts
        self.system_prompt = system_prompt
        self.k = k
        self.memory = deque(maxlen=memory_size)
        self.retriever = self.vdb.chroma_db.as_retriever(
            search_kwargs={"k": self.k}
        )
        retrieval_system_prompt = ChatPromptTemplate([
            ("system", self.system_prompt +  "\n<context>{context}</context>" + "\n<memory>{memory}</memory>"),
            ("human", "{input}")
        ])
        combine_docs_chain = create_stuff_documents_chain(self.rag_llm.model, retrieval_system_prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

    def append_to_memory(self, user_message: str, ai_response: str) -> None:
        '''
        Stores user input and AI response in memory

        Args:
        - user_message (str): The user message
        - ai_response (str): The AI response
        '''
        self.memory.append({"human": user_message.strip(), "ai": ai_response.strip()})
    
    def get_memory_string(self) -> str:
        '''
        Formats memory into a string for the model.

        Returns:
        - str: The memory string
        '''
        return "\n".join([f"Human: {item['human']}\nAI: {item['ai']}" for item in self.memory])
        
    
    def stream(self, user_message: str) -> Generator[str, None, None]:
        '''
        Generates a response while maintaining memory.

        Args:
        - user_message (str): The user message

        Yields:
        - str: The response from the model
        '''
        self.rag_llm.append_human_memory(user_message)
        memory_context = self.get_memory_string()
        # logger.debug(f"memory_context: {memory_context}")

        # Retrieve documents for debugging purposes
        retrieved_documents = self.retriever.invoke(user_message)
        ContextLogger().log_retrieved_documents(user_message ,retrieved_documents)

        result = self.rag_chain.stream({
            "input": user_message,
            "memory": memory_context  
        })
        response = ""
        for chunk in result:
            if "answer" in chunk:
                logger.debug(f"result: {chunk}")
                response += chunk["answer"]
                yield chunk["answer"]
        self.rag_llm.append_ai_memory(response)
        self.append_to_memory(user_message, response)
