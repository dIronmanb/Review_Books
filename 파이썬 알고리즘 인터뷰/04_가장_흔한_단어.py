# -*- coding: utf-8 -*-

# 금지된 단어를 제외한 가장 흔하게 등장하는 단어를 출력하라
# 대소줌ㄴ자 구분 X, 구두표 또한 무시
from typing import List
from collections import Counter

class solution:
    def mostCommonWord(self, s : str, banned : List[str]) -> str:

        word_list = [i for i in s.split()]
        word_list = Counter(word_list)
        for key, value in word_list.items():
            if key in banned:
                del word_list[key]
        
        print(word_list.most_common())
        
        

