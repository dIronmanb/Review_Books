import collections
import re
from typing import List

class solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        
        words = [word for word in 
                re.sub(r'[^\w]', ' ', paragraph).lower().split()
                if word not in banned] # ^\w : 단어 문자가 아닌 모든 것들은 모두 공백으로 치환

        counts = collections.Counter(words)
        return counts.most_common(1)[0][0]   #(1)까지만 쓰면 [('ball', 2)]가 리턴
            
