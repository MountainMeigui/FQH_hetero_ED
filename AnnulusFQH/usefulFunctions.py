import numpy as np


def modifiedGramSchmidt(A):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure
    :param A: a matrix of column vectors
    :return: a matrix of orthonormal column vectors
    """
    # assuming A is a square matrix
    dim = A.shape[1]
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, dim):
        q = A[:, j]
        for i in range(0, j):
            rij = np.vdot(Q[:, i], q)
            q = q - rij * Q[:, i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj, 0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:, j] = q / rjj
    return Q


def truncate_num(number):
    if number >= 1 and number < 10:
        number = int(number)
        return number


    if number < 1:
        scale = 0.1
        digits = 1
        number = 10 * number
        while number < 1:
            number = 10 * number
            scale = scale / 10
            digits = digits + 1

        number = int(number)
        number = number * scale
        number = round(number,digits)
        return number

    if number >= 10:
        scale = 10
        number = number * 0.1
        while number >= 10:
            number = number * 0.1
            scale = scale * 10
        number = int(number)
        number = number * scale
        return number

def is_sparse_matrix_hermitian(matrix):
    hilbert_dim = matrix.shape[0]
    matrix = matrix.tolil()
    for i in range(hilbert_dim):
        for j in range(i):
            if matrix[i, j] != matrix[j, i]:
                print(str((i, j)))
                print(matrix[i, j])
                print(matrix[j, i])
                return False
    return True


def empty_colomns(matrix):
    hilbert_dim = matrix.shape[0]
    print("hilbert space dimension is: " + str(hilbert_dim))
    matrix = matrix.tolil()
    vec_empty = []
    for i in range(hilbert_dim):
        if matrix[i, i] == 0j:
            vec_empty.append(i)
    return vec_empty