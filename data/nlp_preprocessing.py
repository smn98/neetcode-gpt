import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined = positive + negative
        words = [word for sentence in combined for word in sentence.split()]
        unique = sorted(set(words))
        vocab = {word: i+1 for i, word in enumerate(unique)}
        encoded = [torch.tensor([vocab[word] for word in sentence.split()]) for sentence in combined]
        tokens = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True)
        return tokens
