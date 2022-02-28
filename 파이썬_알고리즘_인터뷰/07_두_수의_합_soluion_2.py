from typing import List


class solution:
    def two_sum(self, nums : List[int], target: int) -> List[int]:
        for i, n in enumerate(nums):
            complement = target - n
        
            if complement in nums[i + 1 : ]:
                return [nums.index(n), nums[i + 1:].index(complement + (i + 1))]
                