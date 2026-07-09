from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        res = []

        for num in numbers:
            num_str = str(num)
            tokens = []
            i = 0
            while i < len(num_str):
                match_found = False

                for j in range(len(num_str), i, -1):
                    substring = num_str[i:j]
                    if substring in vocab:
                        tokens.append(substring)
                        i = j
                        match_found = True
                        break

                if not match_found:
                    tokens.append(num_str[i])
                    i += 1
            res.append(tokens)
        return res
        

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        token_count = 0
        i = 0
        while i < len(text):
            match_found = False
            for j in range(len(text), i, -1):
                substring = text[i:j]
                if substring in vocab:
                    token_count += 1
                    i = j
                    match_found = True
                    break
            if not match_found:
                token_count += 1
                i += 1
        return token_count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        if not words:
            return 0.0
        
        token_count = self.count_tokens(text, vocab)
        return round(token_count / len(words), 4)
