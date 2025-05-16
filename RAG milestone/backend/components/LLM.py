from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class LLM:
    '''
    LLM is a class that represents a Language Model with streaming and memory support.
    '''
    def __init__(self, model: ChatOllama, system_prompt: str = "", use_memory: bool = False):
        '''
        Initializes the LLM with a system prompt and a model.

        Args:
        - system_prompt (str): The system prompt 
        - model (ChatOllama): The model that the LLM will use to generate responses
        - use_memory (bool): Whether to use memory or not
        '''
        self.model = model
        self.memory = [SystemMessage(system_prompt)]
        self.use_memory = use_memory


    def invoke(self, user_message: str) -> str:
        '''
        Generates a response while maintaining memory.

        Args:
        - user_message (str): The user message

        Returns:str
        - str: The response from the model
        '''
        if self.use_memory: self.memory.append(HumanMessage(user_message))
        response = self.model.invoke(self.memory if self.use_memory else user_message)
        if self.use_memory: self.memory.append(AIMessage(response))
        return response

    def stream(self, user_message: str):
        '''
        Streams the conversation while maintaining memory.

        Args:
        - user_message (str): The user message

        Yields:
        - str: The response chunks in real-time
        '''
        self.memory.append(HumanMessage(user_message))
        response = ""
        for chunk in self.model.stream(self.memory):
            response += chunk.content
            yield chunk.content
        self.memory.append(AIMessage(response))

    def append_human_memory(self, message: str):
        '''
        Appends a message to the memory.

        Args:
        - message (str): The message to append
        '''
        self.memory.append(HumanMessage(message))

    def append_ai_memory(self, message: str):
        '''
        Appends a message to the memory.

        Args:
        - message (str): The message to append
        '''
        self.memory.append(AIMessage(message))