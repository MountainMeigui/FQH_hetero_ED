import sys
import os
import yaml

file = open('../configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()

if os.getcwd()[0] == '/':
    working_dir = docs['directories']['working_dir']
    sys.path.append(working_dir)

import numpy as np
import random
from time import sleep
from AnnulusFQH import InteractionMatrices as LCA, BasisAndMBNonInteracting as GA, \
    singleParticleOperatorsOnAnnulus as SPA
from DataManaging.ParametersAnnulus import *
from DataManaging import fileManaging as FM
from ATLASClusterInterface import JobSender as JS, MaSWrapperForATLAS as AMASW

script_name = sys.argv[0]
m = 3

matrix_name = sys.argv[1]
MminL = int(sys.argv[2])
MmaxL = int(sys.argv[3])
edge_states = int(sys.argv[4])
N = int(sys.argv[5])
lz_val = sys.argv[6]
matrix_label = sys.argv[7]
slice_start = int(sys.argv[8])
slice_end = int(sys.argv[9])
speeding_parameter = int(sys.argv[10])
params_filename = sys.argv[11]

Mmin = MminL - edge_states
Mmax = MmaxL + edge_states
params = ParametersAnnulus(params_filename)

JS.limit_num_threads()
if params.matrix_pieces_queue == 'P' or params.matrix_pieces_queue == 'M':
    JS.limit_num_threads(4)

"""
matrix_name in [interactions,FM_term,confining_potential,edge_correlation,density]
"""

if lz_val == 'not_fixed':
    basis_list = GA.create_basis_annulus(Mmin, Mmax, N)
else:
    lz_val = int(float(lz_val))
    basis_list = GA.create_basis_annulus_const_lz(Mmin, Mmax, N, lz_val)

state2index = GA.state_index_dict(basis_list)

if matrix_name == 'interactions':
    if matrix_label[:8] == 'toy_flux':
        potential_type = matrix_label[:8]
        magnetic_flux = float(matrix_label[9:])
        H_2_particles = LCA.extract_two_particle_hamiltonian(Mmin, Mmax, potential_type, magnetic_flux)
    else:
        potential_type = matrix_label
        H_2_particles = LCA.extract_two_particle_hamiltonian(Mmin, Mmax, potential_type)

    max_num_nnz = LCA.maximal_num_non_zero_elements_in_col_interacting(Mmin, Mmax, N) * (slice_end - slice_start)
    max_num_nnz = int(max_num_nnz)

    row_total = np.zeros(shape=max_num_nnz, dtype=int)
    col_total = np.zeros(shape=max_num_nnz, dtype=int)
    mat_elements_total = np.zeros(shape=max_num_nnz, dtype=complex)
    index = 0

    for m in range(slice_start, slice_end):
        vec = basis_list[m]
        row, col, mat_elements = LCA.colum_in_interaction_matrix(m, vec, state2index, H_2_particles, Mmin)
        chunk_size = len(row)

        row_total[index:index + chunk_size] = row
        col_total[index:index + chunk_size] = col
        mat_elements_total[index:index + chunk_size] = mat_elements
        index = index + chunk_size

    if index < max_num_nnz:
        row_total = row_total[:index]
        col_total = col_total[:index]
        mat_elements_total = mat_elements_total[:index]


# Now the single particle operator case
else:

    if matrix_name == 'FM_term':
        FM_term_name = matrix_label
        single_particle_operator = SPA.FM_hamiltonian_term_single_particle(Mmin, Mmax, MminL, MmaxL, FM_term_name)

    if matrix_name == 'confining_potential':
        confining_potential_name = matrix_label
        single_particle_operator = SPA.create_confining_potential(Mmin, Mmax, MminL, MmaxL, confining_potential_name)

    if matrix_name == 'total_angular_momentum':
        lz_term_name = matrix_label
        single_particle_operator = SPA.create_single_particle_operator(MminL, MmaxL, edge_states, lz_term_name,
                                                                       matrix_name)

    max_num_nnz = GA.maximal_num_non_zero_per_row_non_interacting(Mmin, Mmax, N) * (slice_end - slice_start)
    max_num_nnz = int(max_num_nnz)

    row_total = np.zeros(shape=max_num_nnz, dtype=int)
    col_total = np.zeros(shape=max_num_nnz, dtype=int)
    mat_elements_total = np.zeros(shape=max_num_nnz, dtype=complex)
    index = 0

    for m in range(slice_start, slice_end):
        vec = basis_list[m]
        row, col, mat_elements = GA.coloum_in_bilinear_operator(m, vec, state2index, single_particle_operator)

        chunk_size = len(row)

        row_total[index:index + chunk_size] = row
        col_total[index:index + chunk_size] = col
        mat_elements_total[index:index + chunk_size] = mat_elements
        index = index + chunk_size

    if index < max_num_nnz:
        row_total = row_total[:index]
        col_total = col_total[:index]
        mat_elements_total = mat_elements_total[:index]


filename_args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]

sleep(random.random())
FM.write_matrix_piece(matrix_name, filename_args, [slice_start, slice_end - 1], row_total, col_total,
                      mat_elements_total)
# sleep(2)
hilbert_space_dim = GA.size_of_hilbert_space(Mmin, Mmax, N, lz_val)
if AMASW.all_pieces_present_in_matrix_pieces_directory(matrix_name, MminL, MmaxL, edge_states, N, lz_val, matrix_label,
                                                       hilbert_space_dim, speeding_parameter):
    AMASW.unite_and_write_full_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, params_filename)
