import sys
import yaml
import os

file = open('configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()

if os.getcwd()[0] == '/':
    working_dir = docs['directories']['working_dir']
    sys.path.append(working_dir)

from AnnulusFQH import singleParticleOperatorsOnAnnulus as SPA, MatricesAndSpectra as AMAS, \
    BasisAndMBNonInteracting as GA, InteractionMatrices as LCA
from DataManaging import generating_parameters_files as genPF, fileManaging as FM, graphData
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt


def plot_single_matrix_values(single_particle_matrix):
    diag = [0] * len(single_particle_matrix)
    for k in range(len(single_particle_matrix)):
        diag[k] = single_particle_matrix[k, k]
    plt.plot(range(len(single_particle_matrix)), diag, '.')
    return diag


def find_basis_terms_from_vecs(vecs, Mmin, Mmax, N):
    basis_list = GA.create_basis_annulus(Mmin, Mmax, N)
    inds = [0] * len(vecs)
    basis_terms = []
    for i in range(len(vecs)):
        inds[i] = np.where(vecs[i] == 1)[0][0]
        basis_terms.append(basis_list[inds[i]])
    return basis_terms, inds


N = 6
# N = 10
MminL = 10
# MmaxL = 19
MmaxL = MminL + N - 1
edge_states = 2
Mmin = MminL - edge_states
Mmax = MmaxL + edge_states

# Mmin = MminL - 1
# Mmax = MmaxL + 1
phi_range = np.arange(0, 1.01, 0.01)
# confining_potential_name = 'space_quadratic_flux'
confining_potential_name = 'space_parabolic_flux'
# confining_potential_name = 'ang_m_lin_phi'
# confining_potential_name = 'exp_flux'
# hamil_lbls = ['None', confining_potential_name, 'None', 'None']
# parameters = [0.0, 1.0, 0.0, 0.0]
# filename = 'flux_IQH_1'
# genPF.create_generic_parameters_file_for_annulus_geo(filename, MminL, MmaxL, edge_states, N, 'not_fixed', 'None', 0.0,
#                                                      confining_potential_name, 1.0, 'None', 0.0)
# full_params_filename = FM.filename_parameters_annulus(filename)
# spectrum = AMAS.get_low_lying_spectrum(MminL, MmaxL, edge_states, N, 'not_fixed', hamil_lbls, parameters,
#                                        full_params_filename, 0)
# phi = 0.1
hilbert_space_dim = GA.size_of_hilbert_space(Mmin, Mmax, N, 'not_fixed')
print(hilbert_space_dim)
FM_term_name = 'spatial_edge_flux'
# FM_term_name = 'spatial33_flux'

full_spec = {}

for phi in phi_range:
    FM_term_single = SPA.FM_hamiltonian_term_single_particle(Mmin, Mmax, MminL, MmaxL, FM_term_name + '_' + str(phi))
    row, col, mat_elements = GA.bilinear_operator_N_particle_subspace(Mmin, Mmax, N, FM_term_single)
    FM_term_multi = sparse.coo_matrix((mat_elements, (row, col)), shape=(hilbert_space_dim, hilbert_space_dim),
                                      dtype=complex)
    FM_term_multi = FM_term_multi.tocsr()

    conf_pot_single = SPA.create_confining_potential(Mmin, Mmax, MminL, MmaxL,
                                                     confining_potential_name + '_' + str(phi))
    row, col, mat_elements = GA.bilinear_operator_N_particle_subspace(Mmin, Mmax, N, conf_pot_single)
    conf_pot_multi = sparse.coo_matrix((mat_elements, (row, col)), shape=(hilbert_space_dim, hilbert_space_dim),
                                       dtype=complex)
    conf_pot_multi = conf_pot_multi.tocsr()
    spec_states = AMAS.calc_eigenVals_Vecs(conf_pot_multi + 0.01 * FM_term_multi, 20)
    # if phi == 0:
    #     vecs = [spec_states[i][1] for i in range(5)]
    #     print([spec_states[i][0] for i in range(5)])
    #     basis_terms, inds = find_basis_terms_from_vecs(vecs, Mmin, Mmax, N)
    #     print(inds)
    #     print(basis_terms)

    spec = [p[0] for p in spec_states]
    full_spec[phi] = spec


# plt.figure()
# conf_pot_single_0 = SPA.create_confining_potential(Mmin, Mmax, MminL, MmaxL,
#                                                    confining_potential_name + '_' + str(0.4))
# plot_single_matrix_values(conf_pot_single_0)
# plt.title('phi=0.4')
# plt.figure()
# conf_pot_single_0 = SPA.create_confining_potential(Mmin, Mmax, MminL, MmaxL,
#                                                    confining_potential_name + '_' + str(0.6))
# plot_single_matrix_values(conf_pot_single_0)



def plot_conf_pot_at_ends():
    plt.figure()
    conf_pot_single_0 = SPA.create_confining_potential(Mmin, Mmax, MminL, MmaxL,
                                                       confining_potential_name + '_' + str(0.0))
    diag0 = plot_single_matrix_values(conf_pot_single_0)

    plt.title('phi=0')
    plt.figure()
    conf_pot_single_1 = SPA.create_confining_potential(Mmin, Mmax, MminL, MmaxL,
                                                       confining_potential_name + '_' + str(1.0))
    diag1 = plot_single_matrix_values(conf_pot_single_1)
    plt.title('phi=1')
    print(diag0)
    print(diag1)
    print([diag1[i] == diag0[i + 1] for i in range(len(diag0) - 1)])

plot_conf_pot_at_ends()
plt.show()


plt.figure()
full_spec, cuttoff_En = graphData.add_cutoff_energy_to_spectrum(full_spec, 250)
for val in full_spec.keys():
    spec_val = np.array(full_spec[val])
    plt.plot([val] * len(spec_val), spec_val, '_')

# plt.figure()
# # plot_single_matrix_values(conf_pot_single)
print(full_spec[0.0][:10])
print(full_spec[1.0][:10])
# # print(diag0)
# # print(diag1)
plt.show()


def check_FM_flux_term_is_correct():
    FM_term_0 = SPA.FM_hamiltonian_term_single_particle(Mmin, Mmax, MminL, MmaxL, FM_term_name)
    FM_term_1 = SPA.FM_hamiltonian_term_single_particle(Mmin, Mmax, MminL, MmaxL, FM_term_name + '_' + str(1.0))

    count = 0
    for i in range(FM_term_0.shape[0] - 1):
        for j in range(FM_term_0.shape[0] - 1):
            if FM_term_0[i + 1, j + 1] != FM_term_1[i, j]:
                count = count + 1
                print("[{},{}]:  ".format(i, j) + str(FM_term_0[i + 1, j + 1]) + " vs. " + str(FM_term_1[i, j]))
    print(str(count) + " out of " + str(FM_term_0.shape[0] ** 2))
    print(np.count_nonzero(FM_term_0))
    print(np.count_nonzero(FM_term_1))
    print("&&&&&&&&&&&&&&&&&&&&&&&")
    return


# check_FM_flux_term_is_correct()



