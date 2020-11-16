import numpy as np

import math
from scipy.special import gamma
from scipy.special import binom
from scipy.special import gammainc
from scipy.special import gammaincc



def create_single_particle_operator(MminL, MmaxL, edge_states, matrix_label, matrix_name):
    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states

    if matrix_name == 'FM_term':
        FM_term_name = matrix_label
        single_particle_operator = FM_hamiltonian_term_single_particle(Mmin, Mmax, MminL, MmaxL, FM_term_name)

    if matrix_name == 'confining_potential':
        confining_potential_name = matrix_label
        single_particle_operator = create_confining_potential(Mmin, Mmax, MminL, MmaxL, confining_potential_name)


    if matrix_name == 'total_angular_momentum':
        single_particle_operator = total_angular_momentum_single_particle(Mmin, Mmax)

    if matrix_name == 'occupation':
        single_particle_operator = single_particle_occupation(MminL, MmaxL, edge_states, matrix_label)

    if matrix_name == 'density':
        r = float(matrix_label)
        single_particle_operator = density_matrix_1_particle_subspace(Mmin, Mmax, r)

    if matrix_name == 'shift_out':
        num_steps = int(matrix_label)
        single_particle_operator = shift_out(Mmin, Mmax, num_steps)

    return single_particle_operator


def FM_hamiltonian_term_single_particle(Mmin, Mmax, MminL, MmaxL, FM_int_name):
    LLL_degeneracy = Mmax - Mmin + 1
    FM_term = np.zeros([LLL_degeneracy, LLL_degeneracy])


    if FM_int_name[:21] == 'alt_spatial_edge_flux':
        phi = 0.0
        if len(FM_int_name) > 21:
            phi = float(FM_int_name[22:])
        deltas = np.arange(MmaxL - MminL, Mmax - Mmin, 1)
        deltas = [int(d) for d in deltas]

        r_min = np.sqrt(2 * Mmin)
        r_max = np.sqrt(2 * Mmax)
        for delta_m in deltas:
            for k in range(LLL_degeneracy):
                m = Mmin + k
                if m + delta_m <= Mmax:
                    term = spatial_FM_matrix_term(m, m + delta_m, phi, r_min, r_max)
                    FM_term[k, k + delta_m] = term
                    FM_term[k + delta_m, k] = term


    if FM_int_name[:14] == 'spatial_fixed1':
        phi = 0.0
        if len(FM_int_name) > 14:
            phi = float(FM_int_name[15:])
        delta_m = MmaxL - MminL + 3
        r_min = np.sqrt(2 * MminL)
        r_max = np.sqrt(2 * MmaxL)
        for k in range(LLL_degeneracy):
            m = Mmin + k
            if m + delta_m <= Mmax:
                term = spatial_FM_matrix_term(m, m + delta_m, phi, r_min, r_max)
                FM_term[k, k + delta_m] = term
                FM_term[k + delta_m, k] = term

    if FM_int_name[:14] == 'spatial_fixed2':
        phi = 0.0
        if len(FM_int_name) > 14:
            phi = float(FM_int_name[15:])
        delta_m = MmaxL - MminL + 4
        r_min = np.sqrt(2 * MminL)
        r_max = np.sqrt(2 * MmaxL)
        for k in range(LLL_degeneracy):
            m = Mmin + k
            if m + delta_m <= Mmax:
                term = spatial_FM_matrix_term(m, m + delta_m, phi, r_min, r_max)
                FM_term[k, k + delta_m] = term
                FM_term[k + delta_m, k] = term

    if FM_int_name[:19] == 'spatial_fixed_mixed':
        phi = 0.0
        if len(FM_int_name) > 19:
            phi = float(FM_int_name[20:])
        deltas = np.arange(MmaxL - MminL + 3, MmaxL - MminL + 7, 1)
        deltas = [int(d) for d in deltas]

        r_min = np.sqrt(2 * MminL)
        r_max = np.sqrt(2 * MmaxL)
        for delta_m in deltas:
            for k in range(LLL_degeneracy):
                m = Mmin + k
                if m + delta_m <= Mmax:
                    term = spatial_FM_matrix_term(m, m + delta_m, phi, r_min, r_max)
                    FM_term[k, k + delta_m] = term
                    FM_term[k + delta_m, k] = term

    return FM_term


def spatial_FM_matrix_term(m, n, phi, r_min, r_max):
    term = 1 / (np.power(2, phi + m / 2.0 + n / 2.0) * np.sqrt(gamma(phi + m + 1) * gamma(phi + n + 1)))
    term = term * np.power(r_min, phi + m + 1) * np.power(r_max, n + phi)
    term = term * math.exp(-(r_max ** 2 + r_min ** 2) / 4)
    return term


def create_confining_potential(Mmin, Mmax, MminL, MmaxL, confining_potential_name):
    m0 = (MminL + MmaxL) / 2
    LLL_degeneracy = Mmax - Mmin + 1

    if confining_potential_name == 'angular_momentum_linear':
        confining_potential = np.zeros([LLL_degeneracy, LLL_degeneracy])
        for k in range(LLL_degeneracy):
            m = k + Mmin
            confining_potential[k, k] = abs(m0 - m)
        return confining_potential


    if confining_potential_name[:19] == 'linear_m_space_flux':
        phi = 0
        if len(confining_potential_name) > 19:
            phi = float(confining_potential_name[20:])
        confining_potential = np.zeros([LLL_degeneracy, LLL_degeneracy])
        for k in range(LLL_degeneracy):
            m = k + Mmin
            r0 = (2 * (m0 + 1)) ** 0.5
            rm = (2 * (phi + m + 1)) ** 0.5
            alpha = 1
            confining_potential[k, k] = alpha * (rm ** 2 - r0 ** 2) * (
                    gammaincc(m + phi + 1, m0 + 1) - gammainc(m + phi + 1, m0 + 1)) + \
                                        alpha * 4 * np.power(m0 + 1, m + phi + 1) * math.exp(-(m0 + 1)) / gamma(
                m + phi + 1)

        return confining_potential

    if confining_potential_name[:25] == 'shift_linear_m_space_flux':
        phi = 0
        if len(confining_potential_name) > 25:
            phi = float(confining_potential_name[26:])
        confining_potential = np.zeros([LLL_degeneracy, LLL_degeneracy])
        for k in range(LLL_degeneracy):
            m = k + Mmin
            r0 = (2 * (m0 + 1 + 1)) ** 0.5
            rm = (2 * (phi + m + 1)) ** 0.5
            alpha = 1
            confining_potential[k, k] = alpha * (rm ** 2 - r0 ** 2) * (
                    gammaincc(m + phi + 1, m0 + 1) - gammainc(m + phi + 1, m0 + 1)) + \
                                        alpha * 4 * np.power(m0 + 1, m + phi + 1) * math.exp(-(m0 + 1)) / gamma(
                m + phi + 1)

        return confining_potential


    print("wrong name")
    return 0


def parabolic_correction(r0, rm, m):
    correction = 2 * rm * r0 * (1 -
                                np.sqrt(math.pi * (m + 1)) * np.power(2.0, -(2 * m + 1)) * binom(2 * m + 1, m))
    return correction


def single_particle_occupation(MminL, MmaxL, edge_states, occupied_orbital):
    Mmax = MmaxL + edge_states
    Mmin = MminL - edge_states

    LLL_degeneracy = Mmax - Mmin + 1
    occupied_orbital = int(occupied_orbital)
    occupation_term = np.zeros([LLL_degeneracy, LLL_degeneracy])
    occupation_term[occupied_orbital - Mmin, occupied_orbital - Mmin] = 1
    return occupation_term


def density_matrix_1_particle_subspace(Mmin, Mmax, z):
    sizeOfHilbertSpace = int(Mmax - Mmin + 1)

    rho = np.zeros([sizeOfHilbertSpace, sizeOfHilbertSpace], dtype=complex)

    for m in range(sizeOfHilbertSpace):
        for n in range(sizeOfHilbertSpace):
            rho[m, n] = GD.eigenfunction_at_point(m + Mmin, z).conjugate() * GD.eigenfunction_at_point(n + Mmin, z)

    return rho


def total_angular_momentum_single_particle(Mmin, Mmax):
    hilbert_space_dim = int(Mmax - Mmin + 1)

    rho = np.zeros([hilbert_space_dim, hilbert_space_dim], dtype=complex)

    for m in range(hilbert_space_dim):
        rho[m, m] = m + Mmin

    return rho


def shift_out(Mmin, Mmax, num_steps):
    hilbert_space_dim = int(Mmax - Mmin + 1)

    shift = np.zeros([hilbert_space_dim, hilbert_space_dim], dtype=complex)
    for m in range(hilbert_space_dim - num_steps):
        # shift[m, m + num_steps] = np.sqrt(2*(m+num_steps))
        shift[m, m + num_steps] = 1

    return shift
