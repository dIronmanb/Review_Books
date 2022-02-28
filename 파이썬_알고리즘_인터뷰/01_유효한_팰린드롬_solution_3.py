import re

class solution:
    def isPalindrome(self, s : str) -> bool:
        s = s.lower()

        # filtering string
        s = re.sub('[^a-z0-9]', '' , s)
        
        # slicing [ : : -1] means 'Inverse string' 
        return s == s[ : :-1] 