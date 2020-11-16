import numpy as np
from AnnulusFQH import BasisAndMBNonInteracting as GA, singleParticleOperatorsOnAnnulus as SPA
import itertools
import matplotlib.pyplot as plt


def create_MB_basis(single_particle_hilbert_sp_size, N):
    which = np.array(list(itertools.combinations(range(single_particle_hilbert_sp_size), N)))

    basis_list = np.zeros((len(which), single_particle_hilbert_sp_size), dtype='int8')
    basis_list[np.arange(len(which))[None].T, which] = 1

    return basis_list


def state_index_dict(basis_list):
    index_dict = dict()
    for i in range(len(basis_list)):
        index_dict[tuple(basis_list[i])] = i
    return index_dict


def get_single_particle_spectrum_states(single_particle_operator):
    vals, vecs = np.linalg.eigh(single_particle_operator)
    # print(vals)
    solution = zip(vals, vecs)
    solution = sorted(solution, key=lambda tup: tup[0], reverse=False)
    vals, vecs = zip(*solution)
    vecs = np.array(vecs)
    # print(vals)
    return vals, vecs


def get_many_body_spectrum_from_single_particle_spectrum(single_particle_spectrum, N):
    single_hilbert_dim = len(single_particle_spectrum)
    basis_list = create_MB_basis(single_hilbert_dim, N)
    MB_hilbert_space_dim = len(basis_list)
    many_body_spectrum = np.zeros(MB_hilbert_space_dim, dtype=float)

    for i in range(MB_hilbert_space_dim):
        energy = sum([single_particle_spectrum[k] for k in range(single_hilbert_dim) if basis_list[i][k]])
        many_body_spectrum[i] = energy

    return many_body_spectrum


def get_observable_expectancy_single(eigenvals, eigenvecs, observable_single):
    hilbert_space_dim = len(eigenvals)
    observable_vals = np.zeros(shape=hilbert_space_dim, dtype=complex)
    for i in range(hilbert_space_dim):
        eigenvec = np.reshape(np.array(eigenvecs[:, i]), newshape=(hilbert_space_dim))
        ob_val = np.transpose(eigenvec.conjugate()) @ observable_single @ eigenvec
        observable_vals[i] = ob_val

    return observable_vals


def get_observable_MB(eigenvals, eigenvecs, observable_single_operator, N):
    single_hilbert_dim = len(eigenvals)
    observable_single_vals = get_observable_expectancy_single(eigenvals, eigenvecs, observable_single_operator)
    basis_list = create_MB_basis(single_hilbert_dim, N)

    MB_hilbert_space_dim = len(basis_list)
    many_body_observable = np.zeros(MB_hilbert_space_dim, dtype=complex)

    for i in range(MB_hilbert_space_dim):
        ob_val = sum([observable_single_vals[k] for k in range(single_hilbert_dim) if basis_list[i][k]])
        many_body_observable[i] = ob_val

    return many_body_observable


def get_occupation_plot(hamiltonian, Mmin, Mmax, only_larger_then_zero=False):
    vals, vecs = get_single_particle_spectrum_states(hamiltonian)
    print(vals)
    occupation_operators = {}
    for m in range(Mmin, Mmax + 1):
        occupation_op = SPA.single_particle_occupation(Mmin, Mmax, 0, m)
        occupation_operators[m] = occupation_op
    occupation_obs_per_eigval = {}
    # for val, vec in zip(vals, vecs):
    #     occupation_obs_per_eigval[val] = np.zeros(shape=(Mmax - Mmin + 1))
    #     for m in range(Mmin, Mmax + 1):
    #         op = occupation_operators[m]
    #         # occupation_obs_per_eigval[val][m - Mmin] = np.transpose(vec.conjugate()) @ op @ vec
    #         occupation_obs_per_eigval[val][m - Mmin] = np.transpose(vec) @ op @ vec

    for i in range(len(vals)):
        # print(vecs)
        vec = vecs[:, i]
        # print(vec)
        # print(vec.shape)
        occupation_obs_per_eigval[vals[i]] = np.zeros(shape=(Mmax - Mmin + 1))
        for m in range(Mmin, Mmax + 1):
            op = occupation_operators[m]
            # occupation_obs_per_eigval[val][m - Mmin] = np.transpose(vec.conjugate()) @ op @ vec
            occupation_obs_per_eigval[vals[i]][m - Mmin] = np.transpose(vec) @ op @ vec
        eps = 10**-16
        if not only_larger_then_zero or abs(vals[i]) >= eps:
            plt.figure()
            plt.plot(list(range(Mmin, Mmax + 1)), occupation_obs_per_eigval[vals[i]], '_')
            plt.title("eigenvalue= " + str(vals[i]))
    return occupation_obs_per_eigval


# single_spec = [10,1,2,3,4,5]
# N=3
# MB_spec = get_body_spectrum_from_single_particle_spectrum(single_spec,N)
# print(MB_spec)
# print(type(MB_spec[0]))
N = 6
MminL = 10
MmaxL = 3 * (N - 1) + MminL
edge_states = 2
Mmin = MminL - edge_states
Mmax = MmaxL + edge_states

# FM_term = SPA.FM_hamiltonian_term_single_particle(Mmin, Mmax, MminL, MmaxL, 'spatial_fixed2')
FM_term = SPA.FM_hamiltonian_term_single_particle(Mmin, Mmax, MminL, MmaxL, 'alt_spatial_edge_flux')
# FM_term = SPA.FM_hamiltonian_term_single_particle(Mmin, Mmax, MminL, MmaxL, 'spatial33_flux')
get_occupation_plot(FM_term, Mmin, Mmax, True)
plt.show()
