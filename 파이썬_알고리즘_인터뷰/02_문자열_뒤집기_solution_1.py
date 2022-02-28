 # -*- coding: utf-8 -*-


class solution:
    def reverseString(self, s : str) -> None:
        # trick!
        s[:] = s[::-1]

        