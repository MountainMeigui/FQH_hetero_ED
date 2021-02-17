import numpy as np

import itertools
import scipy.sparse as sparse
from datetime import datetime
from time import sleep
from random import random

from DataManaging import fileManaging as FM


def create_basis_annulus(Mmin, Mmax, N):
    try:
        basis_list = FM.read_basis_annulus(0, Mmax - Mmin, N)

    except  FileNotFoundError:
        print("must create basis annulus")

        if N == 0:
            basis_list = np.array([[0] * (Mmax - Mmin + 1)])
        else:
            start_time = datetime.now()
            LLL_degeneracy = Mmax - Mmin + 1

            which = np.array(list(itertools.combinations(range(LLL_degeneracy), N)))

            basis_list = np.zeros((len(which), LLL_degeneracy), dtype='int8')
            basis_list[np.arange(len(which))[None].T, which] = 1
            duration = datetime.now() - start_time
            print("creating basis list took: " + str(duration))

        basis_list = np.array(basis_list, dtype=bool)

        FM.write_basis_annulus(0, Mmax - Mmin, N, basis_list)
        FM.write_size_of_hilbert_space(0, Mmax - Mmin, N, 'not_fixed', len(basis_list))

    except EOFError:
        sleep(random())
        basis_list = FM.read_basis_annulus(0, Mmax - Mmin, N)

    return basis_list


def create_basis_annulus_const_lz(Mmin, Mmax, N, lz_val):
    try:
        basis_list = FM.read_basis_annulus_const_lz(0, Mmax - Mmin, N, lz_val - N * Mmin)

    except  FileNotFoundError:
        print("must create basis annulus")

        start_time = datetime.now()
        basis_list_all = create_basis_annulus(Mmin, Mmax, N)
        size_of_mother_hilbert_space = len(basis_list_all)
        lz_vals = np.zeros(size_of_mother_hilbert_space, dtype=int)
        for i in range(size_of_mother_hilbert_space):
            vec = basis_list_all[i]
            lz_vals[i] = sum(np.where(vec == 1)[0]) + N * Mmin
        basis_list = basis_list_all[lz_vals == lz_val]
        duration = datetime.now() - start_time
        print("creating basis list took: " + str(duration))
        basis_list = np.array(basis_list, dtype=bool)

        FM.write_basis_annulus_const_lz(0, Mmax - Mmin, N, lz_val - N * Mmin, basis_list)

        FM.write_size_of_hilbert_space(0, Mmax - Mmin, N, lz_val - N * Mmin, len(basis_list))
    return basis_list


def state_index_dict(basis_list):
    index_dict = dict()
    for i in range(len(basis_list)):
        index_dict[tuple(basis_list[i])] = i
    return index_dict


def fermion_sign_particle_in_multistate(vec, index):
    ones = np.where(vec == 1)[0]
    sign = np.where(ones == index)[0][0]
    sign = (-1) ** sign

    return sign


def size_of_hilbert_space(Mmin, Mmax, N, lz_val):
    """
    Don't forget about lz_val=='not_fixed' convention
    """
    try:
        if lz_val == 'not_fixed':
            hilbert_space_size = FM.read_size_of_hilbert_space(0, Mmax - Mmin, N, lz_val)
        else:
            hilbert_space_size = FM.read_size_of_hilbert_space(0, Mmax - Mmin, N, lz_val - N * Mmin)
        return hilbert_space_size

    except FileNotFoundError:
        if lz_val == 'not_fixed':
            create_basis_annulus(0, Mmax - Mmin, N)
        else:
            create_basis_annulus_const_lz(0, Mmax - Mmin, N, lz_val - N * Mmin)
        sleep(2)
        hilbert_space_size = FM.read_size_of_hilbert_space(Mmin, Mmax, N, lz_val)
        return hilbert_space_size


def buff_vector(m_start, m_end, vec):
    if m_start > 0:
        vec = np.append([False] * m_start, vec)
    if m_end > 0:
        vec = np.append(vec, [False] * m_end)
    return vec


def is_same_basis_vector(vec1, vec2):
    result = np.prod(vec1 == vec2)
    return result


def indices_of_basis1_in_basis2(basis1_params, basis2_params):
    Mmin1 = basis1_params[0]
    Mmax1 = basis1_params[1]
    N1 = basis1_params[2]
    lz1 = basis1_params[3]

    Mmin2 = basis2_params[0]
    Mmax2 = basis2_params[1]
    N2 = basis2_params[2]
    lz2 = basis2_params[3]

    if N1 != N2 or lz1 != lz2:
        print("not valid request! different particle numbers of different lz values!")
        return 0

    basis1 = create_basis_annulus_const_lz(Mmin1, Mmax1, N1, lz1)
    basis2 = create_basis_annulus_const_lz(Mmin2, Mmax2, N2, lz2)
    indices = np.zeros(len(basis1), dtype=int)
    for i in range(len(basis1)):
        vec = basis1[i]
        vec = buff_vector(Mmin1 - Mmin2, Mmax2 - Mmax1, vec)
        for j in range(len(basis2)):
            if is_same_basis_vector(vec, basis2[j]):
                indices[i] = j
                break
    return indices


def convert_vec_from_basis1_to_basis2(vec, basis1_params, basis2_params):
    convert_ind = indices_of_basis1_in_basis2(basis1_params, basis2_params)
    hilbert_dim1 = size_of_hilbert_space(*basis1_params)
    hilbert_dim2 = size_of_hilbert_space(*basis2_params)

    vec_in2 = np.zeros(shape=[hilbert_dim2, 1], dtype=complex)
    for i in range(hilbert_dim1):
        vec_in2[convert_ind[i]] = vec[i]

    return vec_in2


def coloum_in_bilinear_operator(m, vec, state2index, single_particle_operator):
    allowed_states = state2index.keys()

    ones = np.where(vec == 1)[0]
    zeros = np.where(vec == 0)[0]

    numNonZero = len(ones) * len(zeros) + 1
    numNonZero = int(numNonZero)
    row = np.zeros(shape=numNonZero, dtype=int)
    col = np.zeros(shape=numNonZero, dtype=int)
    mat_elements = np.zeros(shape=numNonZero, dtype=complex)

    index = 0

    diag_element = 0

    for o in ones:
        diag_element = diag_element + single_particle_operator[o, o]
        for z in zeros:
            new_vec = np.array(vec)
            new_vec[o] = 0
            new_vec[z] = 1
            if tuple(new_vec) in allowed_states:
                new_vec_index = state2index[tuple(new_vec)]
                sign = fermion_sign_particle_in_multistate(vec, o) * fermion_sign_particle_in_multistate(
                    new_vec, z)

                mat_element = sign * single_particle_operator[z, o]

                if mat_element != 0:
                    row[index] = new_vec_index
                    col[index] = m
                    mat_elements[index] = mat_element
                    index = index + 1

    if diag_element != 0:
        row[index] = m
        col[index] = m
        mat_elements[index] = diag_element
        index = index + 1

    if index < numNonZero:
        row = row[:index]
        col = col[:index]
        mat_elements = mat_elements[:index]

    return row, col, mat_elements


def maximal_num_non_zero_per_row_non_interacting(Mmin, Mmax, N):
    filled_single_particle_states = N
    empty_single_particle_states = Mmax - Mmin + 1 - N

    nnz = filled_single_particle_states * empty_single_particle_states + 1
    return nnz


def bilinear_operator_N_particle_subspace_fixed_lz(Mmin, Mmax, N, single_particle_operator, lz_value):
    basis_list = create_basis_annulus_const_lz(Mmin, Mmax, N, lz_value)

    MB_dim = len(basis_list)

    max_num_operator_entries = MB_dim * maximal_num_non_zero_per_row_non_interacting(Mmin, Mmax, N)
    max_num_operator_entries = int(max_num_operator_entries)

    row_total = np.zeros(shape=max_num_operator_entries, dtype=int)
    col_total = np.zeros(shape=max_num_operator_entries, dtype=int)
    mat_elements_total = np.zeros(shape=max_num_operator_entries, dtype=complex)

    index = 0

    state2index = state_index_dict(basis_list)

    for m in range(MB_dim):
        vec = basis_list[m]

        row, col, mat_elements = coloum_in_bilinear_operator(m, vec, state2index, single_particle_operator)

        chunk_size = len(row)

        row_total[index:index + chunk_size] = row
        col_total[index:index + chunk_size] = col
        mat_elements_total[index:index + chunk_size] = mat_elements
        index = index + chunk_size

    if index < max_num_operator_entries:
        row_total = row_total[:index]
        col_total = col_total[:index]
        mat_elements_total = mat_elements_total[:index]

    return row_total, col_total, mat_elements_total


def bilinear_operator_N_particle_subspace(Mmin, Mmax, N, single_particle_operator):
    basis_list = create_basis_annulus(Mmin, Mmax, N)

    MB_dim = len(basis_list)

    max_num_operator_entries = MB_dim * maximal_num_non_zero_per_row_non_interacting(Mmin, Mmax, N)
    max_num_operator_entries = int(max_num_operator_entries)

    row_total = np.zeros(shape=max_num_operator_entries, dtype=int)
    col_total = np.zeros(shape=max_num_operator_entries, dtype=int)
    mat_elements_total = np.zeros(shape=max_num_operator_entries, dtype=complex)

    index = 0
    state2index = state_index_dict(basis_list)

    for m in range(MB_dim):
        vec = basis_list[m]

        row, col, mat_elements = coloum_in_bilinear_operator(m, vec, state2index, single_particle_operator)
        chunk_size = len(row)

        row_total[index:index + chunk_size] = row
        col_total[index:index + chunk_size] = col
        mat_elements_total[index:index + chunk_size] = mat_elements
        index = index + chunk_size

    if index < max_num_operator_entries:
        row_total = row_total[:index]
        col_total = col_total[:index]
        mat_elements_total = mat_elements_total[:index]

    return row_total, col_total, mat_elements_total


def calc_operator_observable_fixed_lz(Mmin, Mmax, N, lz_val, state, single_particle_operator):
    operator = bilinear_operator_N_particle_subspace_fixed_lz(Mmin, Mmax, N, single_particle_operator, lz_val)
    row_state = np.transpose(state).conjugate()

    ob = row_state @ operator @ state
    return ob


def calc_operator_observable(Mmin, Mmax, N, state, single_particle_operator):
    operator = bilinear_operator_N_particle_subspace(Mmin, Mmax, N, single_particle_operator)
    row_state = np.transpose(state).conjugate()
    ob = row_state @ operator @ state
    return ob
