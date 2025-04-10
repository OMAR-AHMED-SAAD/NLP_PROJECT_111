from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import Lowercase
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD
from tokenizers.trainers import BpeTrainer
from typing import List, Tuple
import os 

class BPETokenizer:
    def __init__(self, unk_token:str="[UNK]", special_tokens: List[str]=["[UNK]", "[PAD]", "[MASK]", "[SOS]", "[EOS]"], tokenizer_dir: str = "./tokenizers", tokenizer_path: str = "tokenizer.json"):
        # check if tokenizer_path exists
        self.special_tokens = special_tokens
        self.unk_token = unk_token
        self.tokenizer_path = tokenizer_path
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer = None
        self.save_dir = f"{self.tokenizer_dir}/{self.tokenizer_path}"

        if not os.path.exists(self.save_dir):
            # Create tokenizer
            print("Creating new tokenizer...")
            os.makedirs(tokenizer_dir, exist_ok=True)
            self.tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
            self.tokenizer.normalizer = Lowercase()
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        else:
            # Load tokenizer
            print(f"Loading tokenizer from {self.save_dir}...")
            self.tokenizer = Tokenizer.from_file(self.save_dir)
            self.tokenizer.post_processor = TemplateProcessing(
                single="[SOS] $A [EOS]",
                special_tokens=[
                    ("[SOS]", self.tokenizer.token_to_id("[SOS]")),
                    ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
                ],
            )
            # Debugging
            unk_id = self.tokenizer.token_to_id(unk_token)
            print(f"Loaded tokenizer. [UNK] token ID: {unk_id}")
            if unk_id is None:
                print(f"Warning: {unk_token} not found in vocabulary!")

    def train(self, combined_text: List[str], vocab_size: int = 10000, min_frequency: int = 2):
        '''
        Train the tokenizer on the combined text.
        
        Args:
            combined_text (List[str]): The list of texts to train on.
        '''

        # Check if tokenizer_path exists
        if os.path.exists(self.save_dir):
            print(f"Tokenizer already exists at {self.save_dir}. Skipping training.")
            
            return
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )
        self.tokenizer.train_from_iterator(combined_text,trainer=trainer)
        self.tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[
                ("[SOS]", self.tokenizer.token_to_id("[SOS]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ],
        )
        self.tokenizer.save(self.save_dir)
        print(f"Tokenizer saved to {self.save_dir}")

    def _check_tokenizer_exists(self) -> bool:
        '''
        Check if the tokenizer exists.
        
        Returns:
            bool: True if the tokenizer exists, False otherwise.
        '''
        return os.path.exists(self.save_dir)

    def encode(self, text: str) -> List[int]:
        '''
        Encode a single text into tokens.
        
        Args:
            text (str): The text to encode.
        '''
        if not self._check_tokenizer_exists():
            raise ValueError(f"Tokenizer does not exist at {self.save_dir}. Please train the tokenizer first.")
        unk_id = self.tokenizer.token_to_id(self.unk_token)
        print(f"Loaded tokenizer. [UNK] token ID: {unk_id}")
        if unk_id is None:
            print(f"Warning: {self.unk_token} not found in vocabulary!")
        tokens = self.tokenizer.encode(text)
        return tokens.ids
    
    def decode(self, tokens: List[int]) -> str:
        '''
        Decode a list of tokens into text.
        
        Args:
            tokens (List[int]): The list of tokens to decode.
        '''
        if not self._check_tokenizer_exists():
            raise ValueError(f"Tokenizer does not exist at {self.save_dir}. Please train the tokenizer first.")
        text = self.tokenizer.decode(tokens)
        return text
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        '''
        Encode a batch of texts into tokens.
        
        Args:
            texts (List[str]): The list of texts to encode.
        '''
        if not self._check_tokenizer_exists():
            raise ValueError(f"Tokenizer does not exist at {self.save_dir}. Please train the tokenizer first.")
        tokens = self.tokenizer.encode_batch(texts)
        return [token.ids for token in tokens]
    
    def decode_batch(self, tokens: List[List[int]]) -> List[str]:
        '''
        Decode a batch of tokens into texts.
        
        Args:
            tokens (List[List[int]]): The list of tokens to decode.
        '''
        if not self._check_tokenizer_exists():
            raise ValueError(f"Tokenizer does not exist at {self.save_dir}. Please train the tokenizer first.")
        texts = self.tokenizer.decode_batch(tokens)
        return texts
    
    


