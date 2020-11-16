import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from time import sleep
from datetime import datetime

from AnnulusFQH import BasisAndMBNonInteracting as GA, InteractionMatrices as LCA, \
    singleParticleOperatorsOnAnnulus as SPA
from AnnulusFQH.usefulFunctions import modifiedGramSchmidt
from DataManaging import fileManaging as FM, graphData
from DataManaging.ParametersAnnulus import *
from ATLASClusterInterface import JobSender as JS, errorCorrectionsAndTests as EC, MaSWrapperForATLAS as AMASW
from clusterScripts import scriptNames
import os
import shutil


def create_matrix_pieces(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, params_filename):
    params = ParametersAnnulus(params_filename)
    common_args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]

    filename_complete_matrix = FM.filename_complete_matrix(matrix_name, common_args)
    if EC.does_file_really_exist(filename_complete_matrix):
        print("already created " + filename_complete_matrix)
        return 0

    queue = params.matrix_pieces_queue
    mem, vmem = JS.get_mem_vmem_vals(queue, params.matrix_pieces_mem, params.matrix_pieces_vmem)

    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states
    hilbert_space_size = GA.size_of_hilbert_space(Mmin, Mmax, N, lz_val)
    if hilbert_space_size < 10000:
        args = [matrix_name] + common_args
        str_args = [str(a) for a in args]
        str_args = '-'.join(str_args)
        filename_job = str_args
        JS.send_job(scriptNames.fullMBMatrix, queue, mem=mem, vmem=vmem, script_args=args,
                    pbs_filename=filename_job)
        return 0

    slice_size = int(hilbert_space_size / params.speeding_parameter)
    last_slice_size = hilbert_space_size - params.speeding_parameter * slice_size

    slice_start = 0
    slice_end = slice_size
    count_num_jobs_sent = 0
    for i in range(params.speeding_parameter):
        filename_args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
        filename_martix_piece = FM.filename_matrix_piece(matrix_name, filename_args, [slice_start, slice_end - 1])
        if EC.does_file_really_exist(filename_martix_piece):
            print("matrix piece " + filename_martix_piece + " already exists!")
        else:
            count_num_jobs_sent = count_num_jobs_sent + 1
            args = [matrix_name] + common_args + [slice_start, slice_end] + [params.speeding_parameter, params_filename]
            str_args = [str(a) for a in args]
            str_args = '-'.join(str_args[:-1])
            filename_job = str_args
            JS.send_job(scriptNames.piecesMBMatrix, queue, mem=mem, vmem=vmem, script_args=args,
                        pbs_filename=filename_job)
            sleep(2)
        slice_start = slice_start + slice_size
        slice_end = slice_end + slice_size
    # taking care of last slice
    if last_slice_size > 0:
        slice_end = slice_start + last_slice_size
        filename_args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
        filename_martix_piece = FM.filename_matrix_piece(matrix_name, filename_args, [slice_start, slice_end - 1])
        if EC.does_file_really_exist(filename_martix_piece):
            print("matrix piece " + filename_martix_piece + " already exists!")
        else:
            count_num_jobs_sent = count_num_jobs_sent + 1
            args = [matrix_name] + common_args + [slice_start, slice_end] + [params.speeding_parameter, params_filename]
            str_args = [str(a) for a in args]
            str_args = '-'.join(str_args[:-1])
            filename = str_args
            JS.send_job(scriptNames.piecesMBMatrix, queue, mem=mem, vmem=vmem, script_args=args,
                        pbs_filename=filename)

    if count_num_jobs_sent == 0:
        AMASW.unite_and_write_full_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name,
                                          params_filename)
    return 0


def unite_matrix_pieces(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, output=1):
    args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    directory = FM.filename_matrix_pieces_directory(matrix_name, args)
    all_sliced_files = [f[:-4].split('_') for f in os.listdir(directory) if f[-4:] == '.npz']
    slices = [[int(f[0]), int(f[2])] for f in all_sliced_files]

    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states
    hilbert_size = GA.size_of_hilbert_space(Mmin, Mmax, N, lz_val)
    if matrix_name == 'interactions':
        max_num_nnz = LCA.maximal_num_non_zero_elements_in_col_interacting(Mmin, Mmax, N) * hilbert_size
        max_num_nnz = int(max_num_nnz)
    else:
        max_num_nnz = GA.maximal_num_non_zero_per_row_non_interacting(Mmin, Mmax, N) * hilbert_size
        max_num_nnz = int(max_num_nnz)

    row_total = np.zeros(shape=max_num_nnz, dtype=int)
    col_total = np.zeros(shape=max_num_nnz, dtype=int)
    mat_elements_total = np.zeros(shape=max_num_nnz, dtype=complex)
    index = 0

    for slice in slices:
        row, col, matrix_elements = FM.read_matrix_piece(matrix_name, args, slice)
        chunk_size = len(row)

        row_total[index:index + chunk_size] = row
        col_total[index:index + chunk_size] = col
        mat_elements_total[index:index + chunk_size] = matrix_elements
        index = index + chunk_size

    if index < max_num_nnz:
        row_total = row_total[:index]
        col_total = col_total[:index]
        mat_elements_total = mat_elements_total[:index]

    FM.write_complete_matrix(matrix_name, args, row_total, col_total, mat_elements_total)
    if output:
        return row_total, col_total, mat_elements_total
    return 0


def delete_excess_pieces(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name):
    args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    directory = FM.filename_matrix_pieces_directory(matrix_name, args)
    filename_full = FM.filename_complete_matrix(matrix_name, args)
    if EC.does_file_really_exist(filename_full):
        shutil.rmtree(directory)
        print("full matrix is present so pieces files were deleted")
        return 0
    print("files were NOT deleted since full matrix file is NOT present")
    return 0


def create_complete_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, output='matrix'):
    args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    filename_matrix = FM.filename_complete_matrix(matrix_name, args)
    if EC.does_file_really_exist(filename_matrix):
        if output:
            row, col, mat_elements = FM.read_complete_matrix(matrix_name, args)
            return row, col, mat_elements
        return 0

    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states
    if matrix_name == 'interactions':
        if matrix_label[:8] == 'toy_flux':
            potential_type = matrix_label[:8]
            magnetic_flux = float(matrix_label[9:])
            H_2_particles = LCA.extract_two_particle_hamiltonian(Mmin, Mmax, potential_type, magnetic_flux)
        else:
            potential_type = matrix_label
            H_2_particles = LCA.extract_two_particle_hamiltonian(Mmin, Mmax, potential_type)

        if lz_val == 'not_fixed':
            row, col, mat_elements = LCA.multi_particle_interaction_energy_matrix(Mmin, Mmax, N, H_2_particles)
        else:
            row, col, mat_elements = LCA.multi_particle_interaction_energy_matrix_const_lz(Mmin, Mmax, N, H_2_particles,
                                                                                           lz_val)

    else:
        single_particle_operator = SPA.create_single_particle_operator(MminL, MmaxL, edge_states, matrix_label,
                                                                       matrix_name)
        if lz_val == 'not_fixed':
            row, col, mat_elements = GA.bilinear_operator_N_particle_subspace(Mmin, Mmax, N, single_particle_operator)
        else:
            row, col, mat_elements = GA.bilinear_operator_N_particle_subspace_fixed_lz(Mmin, Mmax, N,
                                                                                       single_particle_operator, lz_val)
    FM.write_complete_matrix(matrix_name, args, row, col, mat_elements)
    if output:
        return row, col, mat_elements
    return 0


def extract_complete_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, run_on_cluster=1):
    args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    matrix_filename = FM.filename_complete_matrix(matrix_name, args)
    hilbert_space_dim = GA.size_of_hilbert_space(MminL - edge_states, MmaxL + edge_states, N, lz_val)
    if EC.does_file_really_exist(matrix_filename):
        row, col, mat_elements = FM.read_complete_matrix(matrix_name, args)
    else:

        if run_on_cluster:
            print("matrix should have been created by now. you have a bug!")
            return 0
        row, col, mat_elements = create_complete_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name)

    matrix = sparse.coo_matrix((mat_elements, (row, col)), shape=(hilbert_space_dim, hilbert_space_dim),
                               dtype=complex)
    matrix = matrix.tocsr()
    return matrix


def calc_hamiltonian_pieces(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, params_filename, send_jobs=True):
    params = ParametersAnnulus(params_filename)
    hamiltonian_terms_names = ['interactions', 'confining_potential', 'SC_term', 'FM_term']
    all_matrices_filenames = []
    for i in range(len(hamiltonian_labels)):
        if hamiltonian_labels[i] != 'None':
            while EC.how_many_jobs() + params.speeding_parameter > params.max_jobs_in_queue_S:
                sleep(100)
                print("too many jobs in queue!")
            args = [MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels[i]]
            matrix_filename = FM.filename_complete_matrix(hamiltonian_terms_names[i], args)
            all_matrices_filenames.append(matrix_filename)
            if not EC.does_file_really_exist(matrix_filename) and send_jobs:
                create_matrix_pieces(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels[i],
                                     hamiltonian_terms_names[i], params_filename)
    return all_matrices_filenames


def extract_Hamiltonian(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters, run_on_cluster=1):
    # [interactions_label,confining_potential_label,SC_label,FM_label],[interactions_parameter,confining_potential_parameter,SC_parameter,FM_parameter]

    hamiltonian_terms_names = ['interactions', 'confining_potential', 'SC_term', 'FM_term']

    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states
    hilbert_space_dim = GA.size_of_hilbert_space(Mmin, Mmax, N, lz_val)

    hamiltonian = sparse.csr_matrix((hilbert_space_dim, hilbert_space_dim), dtype=complex)

    for i in range(len(hamiltonian_labels)):
        if hamiltonian_labels[i] != 'None':
            args = [MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels[i]]
            filename_ham_term = FM.filename_complete_matrix(hamiltonian_terms_names[i], args)
            if EC.does_file_really_exist(filename_ham_term):
                row, col, mat_elements = FM.read_complete_matrix(hamiltonian_terms_names[i], args)
            else:

                if run_on_cluster:
                    print("matrix should have been created by now. you have a bug!")
                    return 0
                else:
                    row, col, mat_elements = create_complete_matrix(MminL, MmaxL, edge_states, N, lz_val,
                                                                    hamiltonian_labels[i], hamiltonian_terms_names[i])

            mat_elements = parameters[i] * mat_elements

            ham_term = sparse.coo_matrix((mat_elements, (row, col)), shape=(hilbert_space_dim, hilbert_space_dim),
                                         dtype=complex)
            ham_term = ham_term.tocsr()

            hamiltonian = hamiltonian + ham_term
    return hamiltonian


def take_eigenvalue(element):
    return element[0]


def calc_eigenVals_Vecs(matrix, k):
    if matrix.shape[0] >= 6500:
        matrix = matrix.tocsr()

        start_time = datetime.now()
        # eigenVals, eigenVecs = sparse.linalg.eigsh(matrix, which='SA', k=k)
        eigenVals, eigenVecs = eigsh(matrix, which='SA', k=k)
        duration = datetime.now() - start_time
        print("finding eigenvalues took " + str(duration))

        start_time = datetime.now()
        spectrum = [(eigenVals[i], np.array(eigenVecs[:, i])) for i in range(len(eigenVals))]
        spectrum = sorted(spectrum, key=take_eigenvalue)
        duration = datetime.now() - start_time
        print("my weird data structure took " + str(duration))

    else:
        eigenVals, eigenVecs = np.linalg.eigh(matrix.todense())
        spectrum = [(eigenVals[i], np.reshape(np.array(eigenVecs[:, i]), eigenVecs.shape[0])) for i in
                    range(len(eigenVals))]
        spectrum = sorted(spectrum, key=take_eigenvalue)

    return spectrum


def calc_lz_total_for_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters, params_filename,
                               run_on_cluster=1):
    filename_lz_spectrum = FM.filename_spectrum_lz_total_vals(MminL, MmaxL, edge_states, N, hamiltonian_labels,
                                                              parameters)
    if EC.does_file_really_exist(filename_lz_spectrum):
        print("already created " + filename_lz_spectrum)
        return 0

    if lz_val != 'not_fixed':
        lz_val = int(float(lz_val))
    filename_spectrum = FM.filename_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels,
                                                         parameters)
    if EC.does_file_really_exist(filename_spectrum):
        spectrum_eigenstates = FM.read_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels,
                                                            parameters)
    else:
        if run_on_cluster:
            print("spectrum should have been written by now")
            return 0
        spectrum_eigenstates = get_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels,
                                                      parameters, params_filename, run_on_cluster)

    spectrum = np.array([a[0] for a in spectrum_eigenstates])
    lz_total_spectrum_vals = np.zeros(len(spectrum_eigenstates))
    lz_total_matrix = extract_complete_matrix(MminL, MmaxL, edge_states, N, lz_val, 'None', 'total_angular_momentum',
                                              run_on_cluster)
    for i in range(len(spectrum_eigenstates)):
        state = spectrum_eigenstates[i][1]
        lz_total_spectrum_vals[i] = np.conjugate(state).transpose() @ lz_total_matrix @ state

    xlabel = 'Lz total'
    ylabel = 'Energy'
    title = 'low spectrum of system with\nN=' + str(N) + ' MminL=' + str(MminL) + ' MmaxL=' + str(
        MmaxL) + ' edge_states=' + str(edge_states) + '\npotential_type=' + hamiltonian_labels[
                0] + '  ' + str(parameters[0]) + '\nconfining potential=' + hamiltonian_labels[1] + '  ' + str(
        parameters[1]) + '\nFM_term=' + hamiltonian_labels[3] + "  " + str(parameters[3])
    graphData.write_graph_data_to_file(filename_lz_spectrum, lz_total_spectrum_vals, spectrum, title, None, xlabel,
                                       ylabel)

    return lz_total_spectrum_vals, spectrum


def get_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters, params_filename,
                           run_on_cluster=1):
    params = ParametersAnnulus(params_filename)
    if lz_val != 'not_fixed':
        lz_val = int(float(lz_val))
    filename_spectrum = FM.filename_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels,
                                                         parameters)
    if EC.does_file_really_exist(filename_spectrum):
        spectrum = FM.read_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters)
        size_hilbert_space = GA.size_of_hilbert_space(MminL - edge_states, MmaxL + edge_states, N, lz_val)
        if len(spectrum) >= params.num_of_eigstates or size_hilbert_space <= params.num_of_eigstates:
            return spectrum

    hamiltonian = extract_Hamiltonian(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters,
                                      run_on_cluster)
    spectrum = calc_eigenVals_Vecs(hamiltonian, params.num_of_eigstates)
    if params.eigenstates_cutoff:
        print("eigenstates_cutoff")
        eig_spectrum_new1 = spectrum[:params.eigenstates_cutoff]

        eig_spectrum_new2 = [(spectrum[i][0], []) for i in range(params.eigenstates_cutoff, len(spectrum))]
        spectrum = eig_spectrum_new1 + eig_spectrum_new2
    FM.write_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters, spectrum)
    return spectrum


def calc_matrix_in_subspace(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name,
                            filename_spectrum_eigenstates, num_states):
    args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    filename_subspace_matrix = FM.filename_edge_subspace_matrix(matrix_name, num_states, args)

    spec_states = FM.read_spectrum_eigenstates_from_file(filename_spectrum_eigenstates)
    subspace_matrix = np.zeros(shape=(num_states, num_states), dtype=complex)

    matrix_full = extract_complete_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, 0)

    for i in range(num_states):
        for j in range(num_states):
            vi = spec_states[i][1]
            vj = spec_states[j][1]
            vi = np.transpose(vi.conjugate())
            val = vi @ matrix_full @ vj
            subspace_matrix[i, j] = val

    FM.write_edge_subspace_matrix(matrix_name, args, num_states, subspace_matrix)
    return subspace_matrix


def extract_matrix_in_low_lying_subspace(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name,
                                         subspace_size, run_on_cluster=0, params_filename_short='basic_config.yml'):
    args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    filename_subspace_matrix = FM.filename_edge_subspace_matrix(matrix_name, subspace_size, args)
    if EC.does_file_really_exist(filename_subspace_matrix):
        matrix_subspace = FM.read_edge_subspace_matrix(matrix_name, subspace_size, args)
        return matrix_subspace
    hamiltonian_labels = ['toy', 'None', 'None', 'None']
    parameters = [1.0, 0.0, 0.0, 0.0]
    params_filename = FM.filename_parameters_annulus(params_filename_short)
    spectrum_eigenstates = get_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels,
                                                  parameters, params_filename, run_on_cluster)
    matrix_full = extract_complete_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name,
                                          run_on_cluster)
    matrix_subspace = np.zeros(shape=[subspace_size, subspace_size], dtype=complex)
    len_vecs = len(spectrum_eigenstates[1][1])
    vecs_in_matrix = np.zeros(shape=[len_vecs, subspace_size], dtype=complex)
    for i in range(subspace_size):
        vecs_in_matrix[:, i] = spectrum_eigenstates[i][1]

    ortho_vecs = modifiedGramSchmidt(vecs_in_matrix)

    for i in range(subspace_size):
        for j in range(i + 1):
            vec_i = ortho_vecs[:, i]
            vec_j = ortho_vecs[:, j]
            vec_i = np.transpose(vec_i.conjugate())
            val = vec_i @ matrix_full @ vec_j
            matrix_subspace[i, j] = val
            matrix_subspace[j, i] = val.conjugate()
    FM.write_edge_subspace_matrix(matrix_name, args, subspace_size, matrix_subspace)

    return matrix_subspace


def extract_low_lying_subspace_Hamiltonian(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters,
                                           subspace_size, run_on_cluster=0):
    # [interactions_label,confining_potential_label,SC_label,FM_label],
    # [interactions_parameter,confining_potential_parameter,SC_parameter,FM_parameter]

    hamiltonian_terms_names = ['interactions', 'confining_potential', 'SC_term', 'FM_term']

    hamiltonian = np.zeros(shape=[subspace_size, subspace_size], dtype=complex)

    for i in range(1, len(hamiltonian_labels)):
        if hamiltonian_labels[i] != 'None':
            # args = [MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels[i]]
            ham_term = extract_matrix_in_low_lying_subspace(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels[i],
                                                            hamiltonian_terms_names[i], subspace_size, run_on_cluster)
            hamiltonian = hamiltonian + parameters[i] * ham_term

    return hamiltonian


def calc_lz_total_for_low_spectrum_subspace(MminL, MmaxL, edge_states, N, hamiltonian_labels, parameters,
                                            subspace_size, run_on_cluster=0):
    filename_lz_spectrum = FM.filename_lz_total_spectrum_low_lying_subspace(MminL, MmaxL, edge_states, N,
                                                                            hamiltonian_labels, parameters,
                                                                            subspace_size)

    if EC.does_file_really_exist(filename_lz_spectrum):
        return filename_lz_spectrum

    lz_val = 'not_fixed'
    hamiltonian = extract_low_lying_subspace_Hamiltonian(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels,
                                                         parameters, subspace_size, run_on_cluster)

    eigen_vals, eigen_vecs = np.linalg.eigh(hamiltonian)

    spectrum_eigenstates = [(eigen_vals[i], eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    spectrum_eigenstates = sorted(spectrum_eigenstates, key=take_eigenvalue)

    spectrum = np.array([a[0] for a in spectrum_eigenstates])

    lz_total_spectrum_vals = np.zeros(len(spectrum_eigenstates))
    lz_total_matrix = extract_matrix_in_low_lying_subspace(MminL, MmaxL, edge_states, N, lz_val, 'None',
                                                           'total_angular_momentum', subspace_size)

    for i in range(len(spectrum_eigenstates)):
        state = spectrum_eigenstates[i][1]
        lz_total_spectrum_vals[i] = np.conjugate(state).transpose() @ lz_total_matrix @ state

    xlabel = 'Lz total'
    ylabel = 'Energy'
    title = 'low spectrum of system with\nN=' + str(N) + ' MminL=' + str(MminL) + ' MmaxL=' + str(
        MmaxL) + ' edge_states=' + str(edge_states) + '\npotential_type=toy 1.0\nconfining potential=' + \
            hamiltonian_labels[1] + '  ' + str(parameters[1]) + '\nFM_term=' + hamiltonian_labels[3] + "  " + str(
        parameters[3])
    graphData.write_graph_data_to_file(filename_lz_spectrum, lz_total_spectrum_vals, spectrum, title, None, xlabel,
                                       ylabel)

    return lz_total_spectrum_vals, spectrum


def get_groundstate(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters, params_filename,
                    run_on_cluster=1):
    spectrum = get_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters,
                                      params_filename, run_on_cluster)
    gs = spectrum[0][1]
    return gs


def calc_full_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_center_val, hamiltonian_labels, parameters,
                                 window_of_lz, params_filename):
    filename_full_spectrum = FM.filename_full_spectrum(MminL, MmaxL, edge_states, N, window_of_lz, hamiltonian_labels,
                                                       parameters)

    if EC.does_file_really_exist(filename_full_spectrum):
        full_spectrum = FM.read_full_spectrum(MminL, MmaxL, edge_states, N, window_of_lz, hamiltonian_labels,
                                              parameters)
        return full_spectrum

    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states
    lz_total_vals = LCA.find_all_lz_total_values(Mmin, Mmax, N)

    if window_of_lz == 'all':
        lz_min = lz_total_vals[0]
        lz_max = lz_total_vals[-1]

    else:
        lz_min = max(lz_center_val - window_of_lz, lz_total_vals[0])
        lz_max = min(lz_center_val + window_of_lz, lz_total_vals[-1])

    full_spectrum = {}
    for lz in range(lz_min, lz_max + 1):
        print("finding the spectrum for lz=" + str(lz))
        spec_states = get_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz, hamiltonian_labels, parameters,
                                             params_filename, 0)
        spectrum_lz = np.array([spec_states[i][0] for i in range(len(spec_states))])
        full_spectrum[lz] = spectrum_lz

    title = 'low spectrum of system with\nN=' + str(N) + ' MminL=' + str(MminL) + ' MmaxL=' + str(
        MmaxL) + ' edge_states=' + str(edge_states) + '\npotential_type=' + hamiltonian_labels[
                0] + '  ' + str(parameters[0]) + '\nconfining potential=' + hamiltonian_labels[1] + '  ' + str(
        parameters[1]) + '\nLz laughlin at ' + str(lz_center_val)
    xlabel = 'Lz total'
    ylabel = 'Energy'
    graphData.write_spectrum_data_to_file(filename_full_spectrum, full_spectrum, title, xlabel, ylabel)
    return full_spectrum


def count_num_edge_states(MminL, MmaxL, edge_states, N, params_filename_short='basic_config.yml'):
    hamiltonian_labels = ['toy', 'None', 'None', 'None']
    h_parameters = [1.0, 0.0, 0.0, 0.0]
    m = 3
    lz_laughlin = m * N * (N - 1) / 2 + MminL * N
    lz_laughlin = int(lz_laughlin)
    params_filename = FM.filename_parameters_annulus(params_filename_short)
    full_spectrum = calc_full_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_laughlin, hamiltonian_labels,
                                                 h_parameters, 'all', params_filename)
    number_of_edge_states = graphData.count_gs_degeneracies(full_spectrum)
    return number_of_edge_states


def polyval(p, x):
    highest_power = len(p) - 1
    val = sum([p[i] * np.power(x, highest_power - i) for i in range(len(p))])
    return val


def calc_luttinger_parameter_from_full_spectrum(full_spectrum, lz_center_val, N, edge_states):
    first_arc_lz = np.array([lz_center_val + i for i in range(N)])
    first_arc_energy = np.array([full_spectrum[lz][0] for lz in first_arc_lz])
    plt.figure()
    plt.plot(first_arc_lz, first_arc_energy, '_')

    p_first_arc, V_first_arc = np.polyfit(first_arc_lz, first_arc_energy, 2, cov=True)
    omega = p_first_arc[0] * 2 * first_arc_lz[0] + p_first_arc[1]

    polyfitted = np.array([np.polyval(p_first_arc, lz) for lz in first_arc_lz])
    plt.plot(first_arc_lz, polyfitted)

    plt.figure()
    umbrella_lz = np.array(
        [lz_center_val - N * edge_states + N * i for i in range(2 * edge_states + 1)])
    umbrella_energy = np.array([full_spectrum[lz][0] for lz in umbrella_lz])
    umbrella_J = np.array([2 * excitation for excitation in range(-edge_states, edge_states + 1)])
    plt.plot(umbrella_lz, umbrella_energy, '_')

    p_umbrella, V_umbrella = np.polyfit(umbrella_J, umbrella_energy, 2, cov=True)
    polyfitted_u = np.array([np.polyval(p_umbrella, J) for J in umbrella_J])
    plt.plot(umbrella_lz, polyfitted_u)
    omega_J_over_four = p_umbrella[0]
    print("vJ = " + str(omega_J_over_four))

    g = omega_J_over_four * 4 / omega
    print("***********")
    print(g)
    plt.show()
    return g


def calc_luttinger_parameter_from_full_spectrum(params_filename, lz_center_val):
    params = ParametersAnnulus(params_filename)
    window_of_lz = 'all'
    filename_full_spectrum = FM.filename_full_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                       params.N, window_of_lz, params.hamiltonian_labels,
                                                       params.h_parameters)
    if EC.does_file_really_exist(filename_full_spectrum):
        full_spectrum = FM.read_full_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                              params.N, window_of_lz, params.hamiltonian_labels, params.h_parameters)
    else:
        full_spectrum = calc_full_low_lying_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                     params.N, lz_center_val, params.hamiltonian_labels,
                                                     params.h_parameters, window_of_lz, params_filename)

    first_arc_lz = np.array([lz_center_val + i for i in range(params.N)])
    first_arc_energy = np.array([full_spectrum[lz][0] for lz in first_arc_lz])
    omega_s = extract_omega_s_from_first_arc(first_arc_lz, first_arc_energy)

    umbrella_lz = np.array(
        [lz_center_val - params.N * params.edge_states + params.N * i for i in range(2 * params.edge_states + 1)])
    umbrella_energy = np.array([full_spectrum[lz][0] for lz in umbrella_lz])
    umbrella_J = np.array([2 * excitation for excitation in range(-params.edge_states, params.edge_states + 1)])

    omega_J = extract_omega_J_from_umbrella_J(umbrella_J, umbrella_lz, umbrella_energy)

    omega2 = (full_spectrum[lz_center_val + 1][0] - full_spectrum[lz_center_val][0]) / 2 + (
            full_spectrum[lz_center_val - 1][0] - full_spectrum[lz_center_val][0]) / 2
    g2 = omega_J / omega2

    g = omega_J / omega_s
    print("***********")
    print("g calculation = " + str(g))
    print("linear approximation for g = " + str(g2))
    return g


def extract_omega_s_from_first_arc(first_arc_lz, first_arc_energy, plot_graphs_to_check=True):
    p_first_arc, V_first_arc = np.polyfit(first_arc_lz, first_arc_energy, 2, cov=True)
    omega_s = p_first_arc[0] * 2 * first_arc_lz[0] + p_first_arc[1]

    if plot_graphs_to_check:
        plt.figure()
        plt.plot(first_arc_lz, first_arc_energy, '_')
        polyfitted = np.array([np.polyval(p_first_arc, lz) for lz in first_arc_lz])
        plt.plot(first_arc_lz, polyfitted)

    # half_N = int(params.N / 2 )
    # p_first_arc, V_first_arc = np.polyfit(first_arc_lz[:half_N], first_arc_energy[:half_N], 1, cov=True)
    # omega = p_first_arc[0]

    return omega_s


def extract_omega_J_from_umbrella_J(umbrella_J, umbrella_lz, umbrella_energy, plot_graphs_to_check=True):
    p_umbrella, V_umbrella = np.polyfit(umbrella_J, umbrella_energy, 2, cov=True)
    omega_J_over_four = p_umbrella[0]
    omega_J = omega_J_over_four * 4

    if plot_graphs_to_check:
        plt.figure()
        plt.plot(umbrella_lz, umbrella_energy, '_')
        polyfitted_u = np.array([np.polyval(p_umbrella, J) for J in umbrella_J])
        plt.plot(umbrella_lz, polyfitted_u)

    return omega_J


def calc_luttinger_parameter_from_scratch(params_filename, lz_center_val, plot_graphs_to_check=True,
                                          do_spectrum_calculation=True, cutoff_edges=False):
    params = ParametersAnnulus(params_filename)
    window_of_lz = 'all'
    filename_full_spectrum = FM.filename_spectrum_luttinger_parm(params.MminLaughlin, params.MmaxLaughlin,
                                                                 params.edge_states,
                                                                 params.N, window_of_lz, params.hamiltonian_labels,
                                                                 params.h_parameters)

    first_arc_lz = np.array([lz_center_val + i for i in range(params.N)])
    umbrella_lz = np.array(
        [lz_center_val - params.N * params.edge_states + params.N * i for i in range(2 * params.edge_states + 1)])

    if EC.does_file_really_exist(filename_full_spectrum):
        full_spectrum = FM.read_spectrum_luttinger_parm(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                        params.N, window_of_lz, params.hamiltonian_labels,
                                                        params.h_parameters)
    else:
        if not do_spectrum_calculation:
            return 0
        full_spectrum = {}

        lz_vals_to_add_spectrum = np.concatenate((first_arc_lz, umbrella_lz))
        for lz_val in lz_vals_to_add_spectrum:
            spectrum_lz = get_low_lying_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N,
                                                 lz_val, params.hamiltonian_labels, params.h_parameters,
                                                 params_filename, 0)
            full_spectrum[lz_val] = [spectrum_lz[i][0] for i in range(len(spectrum_lz))]

        graphData.write_spectrum_data_to_file(filename_full_spectrum, full_spectrum, 'luttinger partial spectrum',
                                              'Total angular momentum', 'Energy')

    first_arc_energy = np.array([full_spectrum[lz][0] for lz in first_arc_lz])
    omega_s = extract_omega_s_from_first_arc(first_arc_lz, first_arc_energy, plot_graphs_to_check)
    #
    umbrella_energy = np.array([full_spectrum[lz][0] for lz in umbrella_lz])
    umbrella_J = np.array([2 * excitation for excitation in range(-params.edge_states, params.edge_states + 1)])

    if cutoff_edges:
        # edges = min(params.edge_states, 4)
        dif = params.edge_states - 4
        if dif > 0:
            umbrella_J = umbrella_J[dif:-dif]
            umbrella_lz = umbrella_lz[dif:-dif]
            umbrella_energy = umbrella_energy[dif:-dif]

    if params.N == 7 and params.edge_states == 4:
        umbrella_J = umbrella_J[:-1]
        umbrella_lz = umbrella_lz[:-1]
        umbrella_energy = umbrella_energy[:-1]

    omega_J = extract_omega_J_from_umbrella_J(umbrella_J, umbrella_lz, umbrella_energy, plot_graphs_to_check)

    omega2 = (full_spectrum[lz_center_val + 1][0] - full_spectrum[lz_center_val][0])

    g2 = omega_J / omega2

    g = omega_J / omega_s
    # print("***********")
    # print("g calculation = " + str(g))
    # print("linear approximation for g = " + str(g2))
    return g


def create_density_profile(params_filename, plot_graph=0):
    params = ParametersAnnulus(params_filename)
    filename_density = FM.filename_density_profile_groundstate(params.MminLaughlin, params.MmaxLaughlin,
                                                               params.edge_states, params.N, params.hamiltonian_labels,
                                                               params.h_parameters, params.num_measurement_points)
    if EC.does_file_really_exist(filename_density):
        print("file for density already exists")
        if plot_graph:
            rs, density_observable = graphData.plot_graph_from_file(filename_density)
            return rs, density_observable
        rs, density_observable = graphData.read_graph_data_from_file(filename_density)
        return rs, density_observable

    spec_states = get_low_lying_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                         params.N, params.lz_laughlin, params.hamiltonian_labels,
                                         params.h_parameters, params_filename, 0)
    gs = spec_states[0][1]
    rmin = (2 * (params.MminLaughlin - params.edge_states)) ** 0.5
    rmax = (2 * (params.MmaxLaughlin + params.edge_states)) ** 0.5

    rs = np.linspace(rmin, rmax, params.num_measurement_points)

    density_observable = np.zeros(params.num_measurement_points)
    for i in range(params.num_measurement_points):
        density_operator = extract_complete_matrix(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                   params.N, int(params.lz_laughlin), rs[i], 'density', 0)
        density_observable[i] = abs(np.transpose(gs).conjugate() @ density_operator @ gs)

    graphData.write_graph_data_to_file(filename_density, rs, density_observable, 'Radial density vs. radial distance')
    return rs, density_observable


def estimate_confining_potential_energy(params_filename):
    params = ParametersAnnulus(params_filename)
    Mmin = params.MminLaughlin - params.edge_states
    Mmax = params.MmaxLaughlin + params.edge_states
    confining_potential_single = SPA.create_confining_potential(Mmin, Mmax, params.MminLaughlin, params.MmaxLaughlin,
                                                                params.confining_potential_name)
    Laughlin_state_conf_vals = [confining_potential_single[k, k] for k in
                                range(params.edge_states,
                                      params.MmaxLaughlin - params.MminLaughlin + params.edge_states)]
    Laughlin_state_energy = sum(Laughlin_state_conf_vals) / 3
    edge_excitations = confining_potential_single[0, 0] * params.N
    total_confining_potential_energy = Laughlin_state_energy + edge_excitations
    total_confining_potential_energy = total_confining_potential_energy * params.confining_potential_parameter
    return total_confining_potential_energy


def estimate_confining_potential_energy_from_raw_parms(MminL, MmaxL, edge_states, N, confining_potential_name):
    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states
    confining_potential_single = SPA.create_confining_potential(Mmin, Mmax, MminL, MmaxL,
                                                                confining_potential_name)
    Laughlin_state_conf_vals = [confining_potential_single[k, k] for k in
                                range(edge_states, MmaxL - MminL + edge_states)]
    Laughlin_state_energy = sum(Laughlin_state_conf_vals) / 3
    edge_excitations = confining_potential_single[0, 0] * N
    total_confining_potential_energy = Laughlin_state_energy + edge_excitations
    return total_confining_potential_energy


def calc_average_gap(spectrum_filename, num_of_states_before_gap):
    spectrum = FM.read_spectrum_data_from_file(spectrum_filename)
    gaps = np.zeros(shape=(len(spectrum)))
    for i, flux in zip(range(len(gaps)), spectrum.keys()):
        gaps[i] = spectrum[flux][num_of_states_before_gap] - spectrum[flux][num_of_states_before_gap - 1]
    average_gap = sum(gaps) / len(gaps)
    return average_gap


def calc_min_gap(spectrum_filename, num_of_states_before_gap):
    spectrum = FM.read_spectrum_data_from_file(spectrum_filename)
    gaps = np.zeros(shape=(len(spectrum)))
    for i, flux in zip(range(len(gaps)), spectrum.keys()):
        gaps[i] = spectrum[flux][num_of_states_before_gap] - spectrum[flux][num_of_states_before_gap - 1]

    min_gap = min(gaps)
    min_index = gaps.argmin()
    flux = list(spectrum.keys())[min_index]
    print("min flux value is = " + str(flux))
    return min_gap


def calc_spectrum_energy_resolution(spectrum_filename, num_of_states_before_gap):
    spectrum = FM.read_spectrum_data_from_file(spectrum_filename)
    res1 = np.zeros(shape=(len(spectrum) - 1))
    fluxes = list(spectrum.keys())
    # flux_delta = fluxes[1] - fluxes[0]
    for i in range(len(spectrum) - 1):
        res1[i] = spectrum[fluxes[i + 1]][num_of_states_before_gap - 1] - spectrum[fluxes[i]][
            num_of_states_before_gap - 1]
    avg1 = sum(res1) / len(res1)

    return max(res1)
