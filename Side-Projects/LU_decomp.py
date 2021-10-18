#Created in Google Colab in July 2021
#Small project done to understand the LUDecomp function used in libraries such as SciPy
#Current function takes in pre-specified input from compile time, but can be changed to accept user input

import numpy as np

mat_A = np.asarray([[1, 3, 4, 8], [2, 5, 8, 10], [6, 3, 5, 9], [2, 4, 6, 1]])

def LUDecomp(matrix):
  identity = np.zeros_like(matrix).astype(np.float32)

  for i in range(identity.shape[0]):
    identity[i, i] = 1;

  mat_P = np.copy(identity)
  mat_L = np.copy(identity)
  mat_U = matrix.astype(np.float32)

  for i in range(mat_U.shape[0]):
    first_val = mat_U[i, i]

    mat_L_scalar = np.copy(identity)
    mat_U_scalar = np.copy(identity)
    mat_U_scalar[i, i] = 1/first_val
    mat_L_scalar[i, i] = first_val

    mat_L = np.matmul(mat_L, mat_L_scalar)
    mat_U = np.matmul(mat_U_scalar, mat_U)

    for j in range(i, mat_U.shape[0]):
      if (i != j):
        scalar = mat_U[j, i]
        mat_mult = np.copy(identity)
        mat_mult[j, i] = scalar
        mat_mult[j, j] *= -1
        mat_L = np.matmul(mat_L, mat_mult)
        mat_U = np.matmul(mat_mult, mat_U)

  return(mat_L, mat_U)

L, U = LUDecomp(mat_A)

print(np.matmul(L, U))
