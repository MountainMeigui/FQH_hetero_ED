import numpy as np
import itertools

import math
from scipy.special import gamma, binom
import matplotlib.pyplot as plt
from time import sleep
import random

from AnnulusFQH.SparseArray4D import *
from AnnulusFQH import BasisAndMBNonInteracting as GA, pseudopotentialsAndInteractionTerm as PDP
from DataManaging import fileManaging as FM


def interaction_matrix_value_annulus_flux(m1, m2, m3, m4, potential_type, magnetic_flux):
    if m1 + m2 - m3 - m4 != 0:
        return 0
    if potential_type == 'toy_flux_delta':
        potential_strength = 1
        matrix_element = potential_strength / (math.pi * np.power(2, 2 * magnetic_flux + m1 + m2 + 2))
        binoms = binom(m1 + m2 + 2 * magnetic_flux, m1 + magnetic_flux) * binom(m3 + m4 + 2 * magnetic_flux,
                                                                                m3 + magnetic_flux)
        matrix_element = matrix_element * np.sqrt(binoms)

    if potential_type == 'toy_flux':
        potential_strength = 1

        gamma_t = gamma(m1 + m2 + 1 + 2 * magnetic_flux) / np.sqrt(
            gamma(m1 + magnetic_flux + 1) * gamma(m2 + magnetic_flux + 1) *
            gamma(m3 + magnetic_flux + 1) * gamma(m4 + magnetic_flux + 1))

        matrix_element = potential_strength * gamma_t / (math.pi * np.power(2, 2 * magnetic_flux + m1 + m2 - 1))
        matrix_element = matrix_element * ((m1 - m2) * (m3 - m4) / (m1 + m2 + 2 * magnetic_flux + 1) - 3 / 4)

    return matrix_element


def create_two_particle_hamiltonian_annulus_flux(Mmin, Mmax, potential_type, magnetic_flux):
    LLL_degeneracy = Mmax - Mmin + 1
    H_2_particles = SparseArray4D()

    states = np.array([i + Mmin for i in range(LLL_degeneracy)])
    pairs = list(itertools.combinations(states, 2))
    ang_z_total = np.array([sum(p) for p in pairs])

    for l in range(LLL_degeneracy):
        m2 = l + Mmin
        for k in range(l + 1):
            m1 = k + Mmin
            hop_to_index = np.where(ang_z_total == m1 + m2)[0]
            for i in hop_to_index:
                p = list(pairs[i])
                if p[0] > p[1]:
                    p.reverse()
                value_straight = interaction_matrix_value_annulus_flux(m1, m2, p[0], p[1], potential_type,
                                                                       magnetic_flux)
                value_exchange = interaction_matrix_value_annulus_flux(m1, m2, p[1], p[0], potential_type,
                                                                       magnetic_flux)
                value = 2 * value_straight - 2 * value_exchange
                H_2_particles.setValue(m1, m2, p[0], p[1], value)
                H_2_particles.setValue(m1, m2, p[1], p[0], -value)
                H_2_particles.setValue(m2, m1, p[0], p[1], -value)
                H_2_particles.setValue(m2, m1, p[1], p[0], value)

    return H_2_particles


def create_two_particle_hamiltonian_annulus(Mmin, Mmax, Vm):
    LLL_degeneracy = Mmax - Mmin + 1
    H_2_particles = SparseArray4D()

    states = np.array([i + Mmin for i in range(LLL_degeneracy)])
    pairs = list(itertools.combinations(states, 2))
    ang_z_total = np.array([sum(p) for p in pairs])

    for l in range(LLL_degeneracy):
        m2 = l + Mmin
        for k in range(l + 1):
            m1 = k + Mmin
            hop_to_index = np.where(ang_z_total == m1 + m2)[0]
            for i in hop_to_index:
                p = pairs[i]
                value = PDP.matrix_element_per_pseudopotentials(m1, m2, p[0], p[1], Vm)
                H_2_particles.setValue(m1, m2, p[0], p[1], value)
                H_2_particles.setValue(m1, m2, p[1], p[0], -value)
                H_2_particles.setValue(m2, m1, p[0], p[1], -value)
                H_2_particles.setValue(m2, m1, p[1], p[0], value)

    return H_2_particles


def maximal_num_non_zero_elements_in_col_interacting(Mmin, Mmax, N):
    num_pairs_of_filled_states = binom(N, 2)
    # max_empty_available_pair_states = (Mmax - Mmin + 1) / 2
    max_empty_available_pair_states = (Mmax - Mmin + 1) / 2 - N / 2
    max_empty_available_pair_states = round(max_empty_available_pair_states + 1)
    nnz = num_pairs_of_filled_states * max_empty_available_pair_states + 1

    return nnz


def colum_in_interaction_matrix(m, vec, state2index, H_2_particles, Mmin):
    ones = np.where(vec == 1)[0]
    zeros = np.where(vec == 0)[0]

    Mmax = Mmin + len(vec) - 1
    N = len(ones)
    max_nnz = maximal_num_non_zero_elements_in_col_interacting(Mmin, Mmax, N)
    max_nnz = int(max_nnz)

    row = np.zeros(shape=max_nnz, dtype=int)
    col = np.zeros(shape=max_nnz, dtype=int)
    matrix_elements = np.zeros(shape=max_nnz, dtype=complex)

    index = 0

    ones = ones + Mmin
    zeros = zeros + Mmin

    one_pairs = list(itertools.combinations(ones, 2))
    zero_pairs = list(itertools.combinations(zeros, 2))

    ang_z_zeros = [sum(p) for p in zero_pairs]

    diag_element = 0

    for op in one_pairs:
        op_index = np.array(op) - Mmin
        ang_z = sum(op)
        hop_to_indices = np.where(ang_z_zeros == ang_z)[0]
        for i in hop_to_indices:
            zp = zero_pairs[i]
            zp_index = np.array(zp) - Mmin
            hop_vec = np.array(vec)
            hop_vec[op_index] = 0
            hop_vec[zp_index] = 1

            hop_vec_ind = state2index[tuple(hop_vec)]
            f_sign = sum(hop_vec[zp_index[0] + 1:zp_index[1]]) + sum(vec[op_index[0] + 1:op_index[1]])
            f_sign = (-1) ** f_sign

            mat_element = float(H_2_particles.getValue(zp[0], zp[1], op[0], op[1])) * f_sign

            if mat_element != 0:
                row[index] = hop_vec_ind
                col[index] = m
                matrix_elements[index] = mat_element
                index = index + 1

        diag_element = diag_element + float(H_2_particles.getValue(op[0], op[1], op[0], op[1]))

    if diag_element != 0:
        row[index] = m
        col[index] = m
        matrix_elements[index] = diag_element
        index = index + 1

    if index < max_nnz:
        row = row[:index]
        col = col[:index]
        matrix_elements = matrix_elements[:index]

    return row, col, matrix_elements


def multi_particle_interaction_energy_matrix(Mmin, Mmax, N, H_2_particles):
    basis_list = GA.create_basis_annulus(Mmin, Mmax, N)
    print("Size of hilbert space: " + str(len(basis_list)))
    MB_dim = len(basis_list)

    max_num_non_zero_entries = int(maximal_num_non_zero_elements_in_col_interacting(Mmin, Mmax, N) * MB_dim)
    row_total = np.zeros(shape=max_num_non_zero_entries, dtype=int)
    col_total = np.zeros(shape=max_num_non_zero_entries, dtype=int)
    mat_elements_total = np.zeros(shape=max_num_non_zero_entries, dtype=complex)
    index = 0

    state2index = GA.state_index_dict(basis_list)

    for m in range(MB_dim):
        vec = basis_list[m]
        row, col, mat_elements = colum_in_interaction_matrix(m, vec, state2index, H_2_particles, Mmin)
        chunk_size = len(row)

        row_total[index:index + chunk_size] = row
        col_total[index:index + chunk_size] = col
        mat_elements_total[index:index + chunk_size] = mat_elements
        index = index + chunk_size
    print("index = " + str(index))
    print("max estimation = " + str(max_num_non_zero_entries))
    if index < max_num_non_zero_entries:
        row_total = row_total[:index]
        col_total = col_total[:index]
        mat_elements_total = mat_elements_total[:index]

    mat_elements_total = 2 * mat_elements_total

    # V = sparse.coo_matrix((mat_elements_total, (row_total, col_total)), shape=(MB_dim, MB_dim))
    # V = V.tocsr()
    # return 2 * V
    return row_total, col_total, mat_elements_total


def multi_particle_interaction_energy_matrix_const_lz(Mmin, Mmax, N, H_2_particles, lz_val):
    basis_list = GA.create_basis_annulus_const_lz(Mmin, Mmax, N, lz_val)
    state2index = GA.state_index_dict(basis_list)

    MB_dim = len(basis_list)

    max_num_non_zero_entries = int(maximal_num_non_zero_elements_in_col_interacting(Mmin, Mmax, N) * MB_dim)
    row_total = np.zeros(shape=max_num_non_zero_entries, dtype=int)
    col_total = np.zeros(shape=max_num_non_zero_entries, dtype=int)
    mat_elements_total = np.zeros(shape=max_num_non_zero_entries, dtype=complex)
    index = 0

    for m in range(MB_dim):
        vec = basis_list[m]
        row, col, mat_elements = colum_in_interaction_matrix(m, vec, state2index, H_2_particles, Mmin)

        chunk_size = len(row)

        row_total[index:index + chunk_size] = row
        col_total[index:index + chunk_size] = col
        mat_elements_total[index:index + chunk_size] = mat_elements
        index = index + chunk_size

    if index < max_num_non_zero_entries:
        row_total = row_total[:index]
        col_total = col_total[:index]
        mat_elements_total = mat_elements_total[:index]
    mat_elements_total = 2 * mat_elements_total

    return row_total, col_total, mat_elements_total


def find_all_lz_total_values(Mmin, Mmax, N):
    min_val = sum(range(Mmin, Mmin + N))
    max_val = sum(range(Mmax - N + 1, Mmax + 1))
    vals = list(range(min_val, max_val + 1))
    return vals


def cutoff_spectrum(spec, energy_cutoff):
    lz_vals = spec.keys()

    for m in lz_vals:
        small_energies = [e for e in spec[m] if e <= energy_cutoff]
        spec[m] = small_energies

    return spec


def calc_spectrum_degeneracies(spectrum):
    all_eigenvalues = list()
    lz_vals = spectrum.keys()
    count = dict()
    for M in lz_vals:
        temp = [(val, M) for val in spectrum[M]]
        all_eigenvalues = all_eigenvalues + temp
        gs = [val for val in spectrum[M] if round(val, 6) == 0]
        if len(gs) > 0:
            count[M] = len(gs)

    all_eigenvalues = sorted(all_eigenvalues)

    all_eigenvalues = [(round(n[0], 6), n[1]) for n in all_eigenvalues]
    first_eigen_vals = all_eigenvalues[0:25]
    x_axis = [e[1] for e in first_eigen_vals]
    y_axis = [e[0] for e in first_eigen_vals]
    plt.plot(x_axis, y_axis, '.')
    plt.title("lowest lying eigenenergies")
    plt.show()

    return all_eigenvalues, count


def extract_two_particle_hamiltonian(Mmin, Mmax, potential_type, magnetic_flux=0):
    try:
        if potential_type == 'toy_flux':
            H_2_particles = FM.read_two_particle_matrices(Mmin, Mmax, potential_type + "_" + str(magnetic_flux))
        else:
            H_2_particles = FM.read_two_particle_matrices(Mmin, Mmax, potential_type)

    except  FileNotFoundError:

        print("must create two particle hamiltonian")
        if potential_type == 'colomb':
            Vm = PDP.colomb_pseudopotentials(Mmax)
            H_2_particles = create_two_particle_hamiltonian_annulus(Mmin, Mmax, Vm)

        if potential_type == 'toy':
            Vm = PDP.toy_pseudopotentials(Mmax, 3)
            #     Note: THIS CODE IS ONLY GOOD FOR M=3
            H_2_particles = create_two_particle_hamiltonian_annulus(Mmin, Mmax, Vm)
            print("working with hardcore pseudopotentials with m=3")

        if potential_type[:15] == 'screened_colomb':
            screening_length = float(potential_type[16:])
            Vm = PDP.screened_colomb_pseudopotentials(Mmax, screening_length)
            H_2_particles = create_two_particle_hamiltonian_annulus(Mmin, Mmax, Vm)

        if potential_type == 'toy_flux':
            H_2_particles = create_two_particle_hamiltonian_annulus_flux(Mmin, Mmax, potential_type, magnetic_flux)
            potential_type = potential_type + "_" + str(magnetic_flux)

        if potential_type == 'toy_flux_delta':
            H_2_particles = create_two_particle_hamiltonian_annulus_flux(Mmin, Mmax, potential_type, magnetic_flux)

        FM.write_two_particle_matrices(Mmin, Mmax, potential_type, H_2_particles)

    except EOFError:
        sleep(random.random())
        if potential_type == 'toy_flux':
            H_2_particles = FM.read_two_particle_matrices(Mmin, Mmax, potential_type + "_" + str(magnetic_flux))
        else:
            H_2_particles = FM.read_two_particle_matrices(Mmin, Mmax, potential_type)

    return H_2_particles
