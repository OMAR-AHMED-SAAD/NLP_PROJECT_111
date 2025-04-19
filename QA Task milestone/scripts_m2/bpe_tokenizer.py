from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import Lowercase
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
from typing import List, Tuple
import os 

class BPETokenizer:
    def __init__(self, unk_token:str="[UNK]", special_tokens: List[str]=["[UNK]", "[PAD]", "[MASK]", "[SOS]", "[EOS]", "[SEP]"], tokenizer_dir: str = "./tokenizers", tokenizer_path: str = "tokenizer.json", max_length: int = -1):
        # check if tokenizer_path exists
        self.special_tokens = special_tokens
        self.unk_token = unk_token
        self.tokenizer_path = tokenizer_path
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer = None
        self.save_dir = f"{self.tokenizer_dir}/{self.tokenizer_path}"
        self.max_length = max_length

        if not os.path.exists(self.save_dir):
            # Create tokenizer
            print("Creating new tokenizer...")
            os.makedirs(tokenizer_dir, exist_ok=True)
            self.tokenizer = Tokenizer(models.BPE(unk_token=unk_token, end_of_word_suffix='##'))
            self.tokenizer.normalizer = Lowercase()
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        else:
            # Load tokenizer
            print(f"Loading tokenizer from {self.save_dir}...")
            self.tokenizer = Tokenizer.from_file(self.save_dir)
            self._customize_tokenizer()

    def _customize_tokenizer(self):
        '''
        Customize the tokenizer with special tokens and other settings.
        '''
        if self.max_length > 0:
            self.set_max_length(self.max_length)
        self.tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [SEP] $B [EOS]",
            special_tokens=[
                ("[SOS]", self.tokenizer.token_to_id("[SOS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ],
        )
        self.tokenizer.decoder = BPEDecoder(suffix='##')

    def set_max_length(self, max_length: int):
        '''
        Set the maximum length for the tokenizer.
        
        Args:
            max_length (int): The maximum length for the tokenizer.
        '''
        self.max_length = max_length
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(length=self.max_length, pad_id=self.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")

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
            end_of_word_suffix='##'
        )
        self.tokenizer.train_from_iterator(combined_text,trainer=trainer)
        self._customize_tokenizer()
        self.tokenizer.save(self.save_dir)
        print(f"Tokenizer saved to {self.save_dir}")


    def get_tokenizer(self):
        """
        Get the underlying tokenizer object.
        
        Returns:
            Tokenizer: The tokenizer object.
        """
        return self.tokenizer

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
        tokens = self.tokenizer.encode(text)
        attention_mask = [1 if tok != self.tokenizer.token_to_id("[PAD]") else 0 for tok in tokens.ids]
        return tokens.ids, attention_mask
    
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
    
    def encode_texts(self, texts: List[str]) -> Tuple[List[int], List[int]]:
        '''
        Encode a single context, question and answer into tokens.
        
        Args:
            texts (List[str]): The texts to encode.

        Returns:
            Tuple[List[int], List[int]]: The encoded tokens and attention mask.
        '''
        if not self._check_tokenizer_exists():
            raise ValueError(f"Tokenizer does not exist at {self.save_dir}. Please train the tokenizer first.")
        context_question_tokens = self.tokenizer.encode(*texts)
        ids_to_mask = [self.tokenizer.token_to_id("[PAD]"), self.tokenizer.token_to_id("[SEP]")]
        attention_mask = [1 if tok not in ids_to_mask else 0 for tok in context_question_tokens.ids]
        return context_question_tokens.ids, attention_mask


    def encode_two_texts(self, context: str, question: str, other_tokens_to_mask: List[str]=None, is_question_first: bool=False) -> Tuple[List[int], List[int]]:
        '''
        Encode a single context, question and answer into tokens.
        
        Args:
            context (str): The context to encode.
            question (str): The question to encode.
            other_tokens_to_mask (List[str]): List of tokens to mask. Default is None.
            is_question_first (bool): If True, the question is encoded first. Default is False.

        Returns:
            Tuple[List[int], List[int]]: The encoded tokens and attention mask.
        '''
        if not self._check_tokenizer_exists():
            raise ValueError(f"Tokenizer does not exist at {self.save_dir}. Please train the tokenizer first.")
        context_question_tokens = self.tokenizer.encode(context, question)
        ids_to_mask = [self.tokenizer.token_to_id("[PAD]")]
        if other_tokens_to_mask is not None:
            for token in other_tokens_to_mask:
                ids_to_mask.append(self.tokenizer.token_to_id(token))
        ids_to_mask = set(ids_to_mask)

        sep_token_id = self.tokenizer.token_to_id("[SEP]")
        attention_mask_context_question = []
        sep_token_seen = False
        for tok in context_question_tokens.ids:
            if tok == sep_token_id:
                sep_token_seen = True
            if tok not in ids_to_mask:
                if sep_token_seen and is_question_first:
                    attention_mask_context_question.append(1)
                elif sep_token_seen and not is_question_first:
                    attention_mask_context_question.append(0)
                elif not sep_token_seen and is_question_first:
                    attention_mask_context_question.append(0)
                elif not sep_token_seen and not is_question_first:
                    attention_mask_context_question.append(1)
            else:
                attention_mask_context_question.append(0)
        
        return context_question_tokens.ids, attention_mask_context_question
