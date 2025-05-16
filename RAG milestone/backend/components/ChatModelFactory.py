from components import LLM
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI 
from enums import ChatEnums, ChatModelTypes

class ChatModelFactory:
    """Factory to instantiate different chat models dynamically."""

    def __init__(self, config: dict):
        '''
        Initializes the ChatModelFactory with a configuration.
        
        Args:
        - config (dict): The configuration
        '''
        self.config = config

    def create_model(self, model_class: str, model_type: str) -> LLM:
        '''
        Creates a chat model instance based on the specified name.

        Args:
        - model_class (str): The model class
        - model_type (str): The model type

        Returns:
        - LLM: The chat model instance
        '''

        if model_type == ChatModelTypes.CHAT.value:
            model_name = self.config.MODEL_CHAT_NAME


        temperature = 0
        streaming = True
        
        if model_class == ChatEnums.OLLAMA.value:
            curr_chat_model = ChatOllama(
                model=model_name,
                temperature=temperature,
                streaming=streaming
            )
        if model_class == ChatEnums.GOOGLE.value:
            curr_chat_model =  ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                api_key=self.config.GOOGLE_API_KEY,
                max_output_tokens=self.config.GOOGLE_MAX_OUTPUT_TOKENS
            )
        return LLM(model=curr_chat_model)