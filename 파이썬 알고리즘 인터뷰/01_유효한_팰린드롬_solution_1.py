class solution:
    def isPalindrome(self, s : str) -> bool:
        strs = []
        for char in s:
            if char.isalnum():
                strs.append(char.lower())

        # see if it's a palindrome or not.
        while len(strs) > 1:
            if strs.pop(0) != strs.pop():
                return False

        return True
    