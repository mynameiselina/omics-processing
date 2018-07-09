import numpy as np
import math


def ctos_row(k, n):
    return (np.ceil(0.5 * (2*n - 1 -
                    np.sqrt(-8*k + 4*n*n - 4*n - 7)) - 1)).astype(int)


def elem_in_i_rows(i, n):
    return (i * (n - 1 - i) + (i*(i + 1)) * 0.5).astype(int)


def ctos_col(k, i, n):
    return (n - elem_in_i_rows(i + 1, n) + k).astype(int)


def ctos(k, n):
    i = ctos_row(k, n)
    j = ctos_col(k, i, n)
    return i, j


def ctos_v(k_row, k, n):
    rows, cols = ctos(k, n)
    rows = np.delete(rows, np.where(rows == k_row))
    cols = np.delete(cols, np.where(cols == k_row))
    k_cols = np.append(rows, cols)

    return k_cols


# def stoc(i,j, n):
# 	if i > j:
# 		temp = i
# 		i = j
# 		j = temp
#
# 	v = np.array(n*i - (i*(i+1))/2 - i + j -1).astype(int)
# 	return v

def stoc_v(row, n):
    if row < n:
        i = np.repeat(row, n-1)
        j = np.arange(0, n)
        j = np.delete(j, row)
        i_new = np.append(j[:row], i[row:])
        j_new = np.append(i[:row], j[row:])
        c = np.array(n*i_new - (i_new*(i_new+1))/2 -
                     i_new + j_new - 1).astype(int)
    else:
        raise Exception('dimension mismatch')

    return c
