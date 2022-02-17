from typing import List

class solution:
    def reorderLogFiles(self, logs : List[str]) -> List[str]:
        letters, digits = [] , []
        for log in logs:
            # Determine whether second string is digitized or not  
            if log.split()[1].isdigit():
                digits.append(log)
            else:
                letters.append(log)
            
            # condition 3
            letters.sort(key = lambda x : (x.split()[1 : ], x.split()[0]))
            
            # print digits as you entered
            return letters + digits