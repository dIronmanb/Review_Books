import collections
from typing import List

class solution:
    def groupAnagrams(self, strs : List[str])->List[List[str]]:
        anagrams = collections.defaultdict(list)

        for word in strs:
            # Sort and append at dict
            # Key-Sorted word : Value-List including words
            anagrams[''.join(sorted(word))].append(word)
        
        # Take values of anagrams, Convert to list
        return list(anagrams.values())
