 # -*- coding: utf-8 -*-

string = ''.join(char for char in input().lower() if char.isalnum())
length = len(string)
mid = length // 2

for i in range(mid):
    if string[i] != string[length - i - 1]:
        print("false")
print("true")

# 예문
# A man, a plan, a canal : Panama
# race a car