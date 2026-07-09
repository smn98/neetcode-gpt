import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        input_tensor = torch.zeros((batch_size, context_length), dtype=torch.long)
        output_tensor = torch.zeros((batch_size, context_length), dtype=torch.long)

        torch.manual_seed(0)
        max_idx = len(data) - context_length - 1

        for i in range(batch_size):
            idx = torch.randint(0, max_idx + 1, (1,)).item()
            x = data[idx: idx + context_length]
            y = data[idx + 1: idx + context_length + 1]
            input_tensor[i] = x
            output_tensor[i] = y

        return (input_tensor, output_tensor)
