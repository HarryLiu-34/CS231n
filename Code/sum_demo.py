import numpy as np
A = np.array([[1,2,3,4], [5,6,7,8]])
B = np.array([1,1,1,1])
print(np.sum(np.abs(A - B), axis = 1))  # [6, 22], row-sum
print(np.sum(np.abs(A - B), axis = 0))  # [4, 6, 8, 10], column-sum
print(A[0, :])