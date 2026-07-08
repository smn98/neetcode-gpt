from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        merges = []
        chars = list(corpus)
        for i in range(num_merges):
            pair_counts = {}
            for j in range(len(chars) - 1):
                pair = (chars[j], chars[j+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            freq_pair = max(sorted(pair_counts), key=pair_counts.get)

            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == freq_pair[0] and chars[i+1] == freq_pair[1]:
                    new_chars.append(freq_pair[0] + freq_pair[1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1

            chars = new_chars

            merges.append([freq_pair[0], freq_pair[1]])
        
        return merges