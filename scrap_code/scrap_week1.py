import numpy as np

def recursive(number):
    if number == 0: return 2
    elif number == 1: return 1
    else: return recursive(number - 1) + recursive(number - 2)

# print(recursive(32))

twoDList = ([1,2,3],[4,5,6],[7,8,9])
matrix = np.array(twoDList)

print(matrix)
print(twoDList)