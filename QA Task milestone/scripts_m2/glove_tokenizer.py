import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, vocab
import torch
import re
torchtext.disable_torchtext_deprecation_warning()


class GloveTokenizer:

    def __init__(
        self,
        ver: str = "6B",
        dim: int = 300,
        special_tokens: list[str] = [
            "[UNK]", "[PAD]", "[MASK]", "[SOS]", "[EOS]", "[SEP]"
        ],
        max_length: int = None,
    ):
        """
        Initialize the GloVe tokenizer.

        Args:
            ver (str): The name of the GloVe model to use. Default is "6B".
            dim (int): The dimensionality of the GloVe vectors. Default is 300.
            special_tokens (list[str]): A list of special tokens to add to the vocabulary.
            max_length (int): The maximum length for the tokenizer. Default is None.
        """

        glove = GloVe(name=ver, dim=dim)
        glove_vocab = vocab(glove.stoi)
        for i, token in enumerate(special_tokens):
            glove_vocab.insert_token(token, i)
        glove_vocab.set_default_index(glove_vocab["[UNK]"])
        pretrained_embeddings = glove.vectors
        pretrained_embeddings = torch.cat(
            (torch.randn(len(special_tokens), pretrained_embeddings.shape[1]),
             pretrained_embeddings))

        self.special_tokens = special_tokens
        self.pretrained_embeddings = pretrained_embeddings
        self.tokenizer = get_tokenizer("basic_english")
        self.glove_vocab = glove_vocab
        self.max_length = max_length
        self.pad_idx = glove_vocab["[PAD]"]

    def get_pretrained_embeddings(self) -> torch.Tensor:
        """
        Get the pretrained GloVe embeddings.

        Returns:
            torch.Tensor: The pretrained GloVe embeddings.
        """
        return self.pretrained_embeddings

    def set_max_length(self, max_length: int):
        """
        Set the maximum length for the tokenizer.

        Args:
            max_length (int): The maximum length for the tokenizer.
        """
        self.max_length = max_length

    def tokenize(self, text1: str, text2: str = None) -> list[str]:
        """
        Tokenize one or two input strings, add special tokens, and PAD to max_length if set.

        Args:
            text1 (str): The first input string.
            text2 (str, optional): The second input string (for paired input).

        Returns:
            list[str]: The tokenized and padded list with special tokens.
        """
        tokens = ["[SOS]"] + self.tokenizer(text1)

        if text2:
            tokens += ["[SEP]"] + self.tokenizer(text2)

        tokens.append("[EOS]")

        if self.max_length:
            tokens = tokens[:self.max_length]
            tokens += ["[PAD]"] * max(0, self.max_length - len(tokens))

        return tokens

    def encode(
        self,
        text: str,
        other_tokens_to_mask: list[str] = []
    ) -> tuple[torch.Tensor, list[int], list[tuple[int, int]]]:
        tokens = self.tokenize(text)  # list of tokens
        token_ids = [self.glove_vocab[token] for token in tokens]
        #  add pad to start of the list
        masked_token_ids = [
            self.glove_vocab[token] for token in other_tokens_to_mask
        ]
        attention_mask = [
            0 if token_id in masked_token_ids else 1 for token_id in token_ids
        ]

        # --- Manually compute offsets ---
        offsets = []
        current_pos = 0
        for token in tokens:
            # Find the next occurrence of the token
            match = re.search(re.escape(token), text[current_pos:])
            if match:
                start = current_pos + match.start()
                end = current_pos + match.end()
                offsets.append((start, end))
                current_pos = end  # move past the current token
            else:
                offsets.append(
                    (0, 0)
                )  # fallback in case not found (shouldn't happen normally)

        return torch.tensor(token_ids), attention_mask, offsets

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode a list of token IDs into text.

        Args:
            token_ids (torch.Tensor): The list of token IDs to decode.
        """
        # Convert token IDs to tokens using the vocab's lookup_tokens method
        #  turn token_ids into a list if it is a tensor
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens = self.glove_vocab.lookup_tokens(token_ids)

        # Remove special tokens
        tokens = [
            token for token in tokens if token not in self.special_tokens
        ]

        return " ".join(tokens)

    def encode_batch(
            self, texts: list[str]) -> tuple[list[list[int]], list[list[int]]]:
        """
        Encode a batch of texts into token IDs and attention masks.

        Args:
            texts (list[str]): The list of texts to encode.

        Returns:
            tuple: A tuple containing:
                - token_ids: List of lists of token IDs.
                - attention_masks: List of lists with 1s for real tokens and 0s for padding.
        """

        token_lists = [self.tokenize(text) for text in texts]
        token_ids = [[self.glove_vocab[token] for token in tokens]
                     for tokens in token_lists]
        attention_masks = [[
            0 if token_id == self.pad_idx else 1 for token_id in ids
        ] for ids in token_ids]

        return token_ids, attention_masks

    def decode_batch(self, token_ids_batch: list[torch.Tensor]) -> list[str]:
        """
        Decode a batch of token IDs into texts.

        Args:
            token_ids_batch (list[torch.Tensor]): A list of token ID tensors to decode.
        """
        # Convert each batch of token IDs to tokens using the vocab's lookup_tokens method
        texts = [
            self.glove_vocab.lookup_tokens(token_ids)
            for token_ids in token_ids_batch
        ]

        # Remove special tokens
        texts = [[token for token in text if token not in self.special_tokens]
                 for text in texts]

        # Join tokens into strings for each batch
        return [" ".join(text) for text in texts]

    def encode_two_texts(
        self,
        context: str,
        question: str,
        other_tokens_to_mask: list[str] = []
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single context and question into tokens.
        Args:
            context (str): The context to encode.
            question (str): The question to encode.
        """
        tokens = self.tokenize(context, question)
        token_ids = [self.glove_vocab[token] for token in tokens]
        #  add pad to start of the list
        masked_token_ids = [
            self.glove_vocab[token] for token in other_tokens_to_mask
        ]
        attention_mask = [
            0 if token_id in masked_token_ids else 1 for token_id in token_ids
        ]

        return torch.tensor(token_ids), attention_mask

    def encode_two_texts_batch(
            self, contexts: list[str], questions: list[str]
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """
        Encode a batch of contexts and questions into token IDs and attention masks.

        Args:
            contexts (list[str]): The list of contexts to encode.
            questions (list[str]): The list of questions to encode.

        Returns:
            tuple: token IDs tensors list, attention masks list
        """
        token_ids = []
        attention_masks = []

        for context, question in zip(contexts, questions):
            tokens = self.tokenize(context, question)
            ids = [self.glove_vocab[token] for token in tokens]
            mask = [0 if token_id == self.pad_idx else 1 for token_id in ids]
            token_ids.append(torch.tensor(ids))
            attention_masks.append(mask)

        return token_ids, attention_masks
