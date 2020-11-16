import numpy as np
from scipy.special import binom
from scipy.special import hyp2f1
from scipy.special import hyp1f1
from scipy.special import gamma

import math


def overlap_two_particle_coordinates_to_relative_and_CM(m1, m2, M, m):
    # print([m1,m2,M,m])
    if m1 + m2 != M + m:
        return 0
    # Since we're working with fermions
    if m % 2 == 0:
        return 0
    if m1 - m + 1 > 0 and m2 - m + 1 > 0:
        # if 0:
        alpha = np.sqrt(gamma(m + 1) * gamma(m1 + m2 - m + 1)) * (
                binom(m1, m) * hyp2f1(-m2, -m, m1 - m + 1, -1) - binom(m2, m) * hyp2f1(-m1, -m, m2 - m + 1, -1))
    else:
        s = sum([np.power(-1, l) * (binom(m1, m - l) * binom(m2, l) - binom(m2, m - l) * binom(m1, l)) for l in
                 range(m + 1)])
        alpha = np.sqrt(gamma(m + 1) * gamma(m1 + m2 - m + 1)) * s
    overlap = alpha / (2 * np.power(2, m1 / 2 + m2 / 2 + 1) * np.sqrt(gamma(m1 + 1) * gamma(m2 + 1)))

    if np.isnan(overlap):
        overlap = 0

    return overlap



def matrix_element_per_pseudopotentials(m1, m2, m3, m4, Vm):
    if m1 + m2 != m3 + m4:
        return 0

    element = 0
    Mrange = [m for m in range(m1 + m2 + 1) if Vm[m] != 0]
    # for m in range(m1 + m2 + 1):
    for m in Mrange:
        M = m1 + m2 - m
        overlap12 = overlap_two_particle_coordinates_to_relative_and_CM(m1, m2, M, m)
        overlap34 = overlap_two_particle_coordinates_to_relative_and_CM(m3, m4, M, m)

        element = element + Vm[m] * overlap12 * overlap34
    return element


def colomb_pseudopotentials(Mmax):
    Vm = np.zeros(int(2 * Mmax))

    for m in range(len(Vm)):
        # Vm[m]=1/(8*math.pi)*gamma(m+0.5)/math.factorial(m)
        Vm[m] = 1 / (8 * math.pi) * gamma(m + 0.5) / gamma(m + 1)
        # Vm[m] = 1 / (8 * math.pi) * gamma(3 / 2) / (m + 3 / 2) * binom(m + 1 / 2, m)
    whereNan = np.isnan(Vm)
    Vm[whereNan] = 0
    # Vm[m] = 1 / (4 * math.pi) * np.power(math.pi, 0.5) /(np.power(2, m+1) * math.factorial(m))* factorial2(2 * m - 1)
    # Vm[m] = math.pi ** 0.5 / 2 * factorial2(2 * m - 1) / (2 ** m * math.factorial(m))*1/(4*math.pi)
    # Vm[m] = math.pi ** 0.5 / 2 * factorial2(2 * m - 1) / (2 ** m * math.factorial(m))

    return Vm


def screened_colomb_pseudopotentials(Mmax, screening_length):
    Vm = np.zeros(int(2 * Mmax))
    alpha = 1 / screening_length
    for m in range(len(Vm)):
        # Vm[m]=1/(8*math.pi)*gamma(m+0.5)/math.factorial(m)
        Vm[m] = 1 / (8 * math.pi) * gamma(m + 0.5) / gamma(m + 1) * hyp1f1(m + 1 / 2, 1 / 2, alpha ** 2)
        # Vm[m] = 1 / (8 * math.pi) * gamma(1.5) / (m + 1.5) * binom(m + 0.5, m) * hyp1f1(m + 1 / 2, 1 / 2,
        #                                                                                       alpha ** 2)
        Vm[m] = Vm[m] - 1 / (4 * math.pi) * alpha * hyp1f1(m + 1, 3 / 2, alpha ** 2)
    print(Vm)
    whereNan = np.isnan(Vm)
    Vm[whereNan] = 0
    # Vm[m] = 1 / (4 * math.pi) * np.power(math.pi, 0.5) /(np.power(2, m+1) * math.factorial(m))* factorial2(2 * m - 1)
    # Vm[m] = math.pi ** 0.5 / 2 * factorial2(2 * m - 1) / (2 ** m * math.factorial(m))*1/(4*math.pi)
    # Vm[m] = math.pi ** 0.5 / 2 * factorial2(2 * m - 1) / (2 ** m * math.factorial(m))

    return Vm


def toy_pseudopotentials(Mmax, m):
    Vm = np.zeros(int(2 * Mmax))
    i = 0
    while i < m:
        if i % 2 != 0:
            Vm[i] = 1
        i = i + 1

    return Vm

