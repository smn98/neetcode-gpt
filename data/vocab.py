from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = sorted(list(set(text)))

        vocab = {char: i for i, char in enumerate(chars)}

        reverse_map = {i: char for char, i in vocab.items()}
        return vocab, reverse_map
        

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        decoded = []
        for char in text:
            decoded.append(stoi[char])
        return decoded


    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        chars = []
        for id in ids:
            chars.append(itos[id])
        return "".join(chars)
