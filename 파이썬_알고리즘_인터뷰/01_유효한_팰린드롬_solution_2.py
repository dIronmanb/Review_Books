import collections
from typing import Deque

class solution:
    def isPalindrome(self, s : str) -> bool:
        # Declare type 'deque'
        strs: Deque = collections.deque()

        for char in s:
            if char.isalnum():
                strs.append(char.lower())

        # Reduce Time by using deque(O(n) -> O(1))
        # pop(0) vs popleft()
        #   O(n)         O(1)
        while len(strs) > 1:
            if strs.popleft() != str.pop():
                return False

        return True 