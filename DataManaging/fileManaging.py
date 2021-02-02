import os
import pickle
import numpy as np
from time import sleep
import random
from ATLASClusterInterface import errorCorrectionsAndTests as EC
import yaml

file = open('../configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()
directories = docs['directories']


def get_storage_dir():
    cwd = os.getcwd()
    if cwd[0] == 'D':
        storage_dir = directories['storage_dir_office']
    if cwd[0] == 'C':
        storage_dir = directories['storage_dir_laptop']
    if cwd[0] == '/':
        storage_dir = directories['storage_dir_ATLAS']
    return storage_dir


"""
In case access to file is problematic
"""


def random_read_helper(func, params):
    try:
        return func(*params)
    except EOFError:
        sleep(random.random())
        return func(*params)


"""
Basic building blocks (basis, interaction of 2 particles)
"""


def filename_two_particle_matrices(Mmin, Mmax, potential_type):
    directory = 'pkl_data/two_particle_hamiltonian_annulus/'
    filename = 'Annulus_' + potential_type + '_two_particle_hamiltonian_Mmin=' + str(Mmin) + '_Mmax=' + str(
        Mmax) + '.pkl'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = directory + filename

    return filename


def directory_two_particle_matrices():
    directory = 'pkl_data/two_particle_hamiltonian_annulus/'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    return directory


def write_two_particle_matrices(Mmin, Mmax, potential_type, H_2_particles):
    filename = filename_two_particle_matrices(Mmin, Mmax, potential_type)
    file1 = open(filename, 'wb')
    dic = {"two_particle_hamiltonian": H_2_particles}
    pickle.dump(dic, file1)
    file1.close()
    print("wrote two_particle_hamiltonian into file: " + filename)
    return 0


def read_two_particle_matrices(Mmin, Mmax, potential_type):
    filename = filename_two_particle_matrices(Mmin, Mmax, potential_type)
    file1 = open(filename, 'rb')
    dic = pickle.load(file1)
    file1.close()
    H_2_particles = dic["two_particle_hamiltonian"]
    print("read two_particle_hamiltonian out of: " + filename)
    return H_2_particles


def filename_basis_annulus(Mmin, Mmax, N):
    filename = 'basis_annulus_Mmin=' + str(Mmin) + '_Mmax=' + str(Mmax) + '_N=' + str(N) + '.npz'
    directory = 'pkl_data/basis_annulus/' + str(N) + '_particles/'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = directory + filename

    return filename


def read_basis_annulus(Mmin, Mmax, N):
    filename = filename_basis_annulus(Mmin, Mmax, N)
    # filename = filename_basis_annulus(0, Mmax - Mmin, N)
    basis_list = np.load(filename)['basis_list']
    return basis_list


def write_basis_annulus(Mmin, Mmax, N, basis_list):
    filename = filename_basis_annulus(Mmin, Mmax, N)
    # filename = filename_basis_annulus(0, Mmax - Mmin, N)
    np.savez(filename, basis_list=basis_list)
    print("wrote basis_list into file: " + filename)
    return 0


def filename_basis_annulus_const_lz(Mmin, Mmax, N, lz_val):
    directory = 'pkl_data/basis_annulus/' + str(N) + '_particles/'
    filename = 'basis_annulus_Mmin=' + str(Mmin) + '_Mmax=' + str(Mmax) + '_N=' + str(N) + '_lz_val=' + str(
        lz_val) + '.npz'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = directory + filename

    return filename


def read_basis_annulus_const_lz(Mmin, Mmax, N, lz_val):
    filename = filename_basis_annulus_const_lz(Mmin, Mmax, N, lz_val)
    basis_list = np.load(filename)['basis_list']
    return basis_list


def write_basis_annulus_const_lz(Mmin, Mmax, N, lz_val, basis_list):
    filename = filename_basis_annulus_const_lz(Mmin, Mmax, N, lz_val)
    np.savez(filename, basis_list=basis_list)
    print("wrote basis_list into file: " + filename)
    return 0


def directory_sizes_of_hilbert_space():
    directory = 'pkl_data/basis_annulus/sizes_of_hilbert_space/'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def filename_size_of_hilbert_space(Mmin, Mmax, N, lz_val):
    directory = directory_sizes_of_hilbert_space() + str(N) + '_particles/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = str(Mmin) + '_' + str(Mmax) + '_' + str(N) + '_' + str(lz_val) + '.npz'

    filename = directory + filename
    return filename


def read_size_of_hilbert_space(Mmin, Mmax, N, lz_val):
    filename = filename_size_of_hilbert_space(Mmin, Mmax, N, lz_val)
    hilbert_space_size = np.load(filename)['hilbert_space_size']
    return hilbert_space_size


def write_size_of_hilbert_space(Mmin, Mmax, N, lz_val, hilbert_space_size):
    filename = filename_size_of_hilbert_space(Mmin, Mmax, N, lz_val)
    np.savez(filename, hilbert_space_size=hilbert_space_size)
    print("wrote size of hilbert space into: " + filename)

    return 0


"""
Spectrum
"""


def filename_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters):
    common_args = [MminL, MmaxL, edge_states, N, lz_val]
    common_args = [str(a) for a in common_args]
    filename = 'low_lying_spectrum_' + '_'.join(common_args) + '_'

    hamiltonian_args = [str(hamiltonian_labels[i]) + '_' + str(parameters[i]) for i in range(len(hamiltonian_labels))]
    filename = filename + '_'.join(hamiltonian_args) + '.pkl'
    directory = 'pkl_data/low_lying_spectrum_annulus/' + str(N) + '_particles/'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = directory + filename
    return filename


def read_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters):
    filename = filename_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters)
    spectrum = np.load(filename)['spectrum']

    return spectrum


def write_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters, spectrum):
    filename = filename_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters)
    np.savez(filename, spectrum=spectrum)
    print("wrote spectrum to file " + filename)
    return 0


def directory_full_spectrum(MminL, MmaxL, edge_states, N):
    directory = 'pkl_data/full_spectrum_annulus/' + str(N) + '_particles/'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)

    sub_dir = '_'.join([str(MminL), str(MmaxL), str(edge_states)]) + '/'
    directory = directory + sub_dir
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def filename_full_spectrum(MminL, MmaxL, edge_states, N, window_of_lz, hamiltonian_labels, parameters):
    # window_of_lz = [number,'all','not_conserved']

    hamiltonian_args = [str(hamiltonian_labels[i]) + '_' + str(parameters[i]) for i in range(len(hamiltonian_labels))]
    args = [MminL, MmaxL, edge_states, N, 'Ham_lbls', *hamiltonian_args, 'lz_win', window_of_lz]
    args = [str(a) for a in args]
    filename = 'full_spectrum_' + '_'.join(args) + '.pkl'
    directory = directory_full_spectrum(MminL, MmaxL, edge_states, N)
    filename = directory + filename
    return filename


def read_full_spectrum(MminL, MmaxL, edge_states, N, window_of_lz, hamiltonian_labels, parameters):
    filename = filename_full_spectrum(MminL, MmaxL, edge_states, N, window_of_lz, hamiltonian_labels, parameters)
    file1 = open(filename, 'rb')
    data = pickle.load(file1)
    file1.close()

    spectrum = data['spectrum']
    return spectrum


def filename_spectrum_luttinger_parm(MminL, MmaxL, edge_states, N, window_of_lz, hamiltonian_labels, parameters):
    # window_of_lz = [number,'all','not_conserved']

    hamiltonian_args = [str(hamiltonian_labels[i]) + '_' + str(parameters[i]) for i in range(len(hamiltonian_labels))]
    args = [MminL, MmaxL, edge_states, N, 'Ham_lbls', *hamiltonian_args, 'lz_win', window_of_lz]
    args = [str(a) for a in args]
    filename = 'luttinger_parm_spectrum_' + '_'.join(args) + '.pkl'
    directory = directory_full_spectrum(MminL, MmaxL, edge_states, N)
    filename = directory + filename
    return filename


def read_spectrum_luttinger_parm(MminL, MmaxL, edge_states, N, window_of_lz, hamiltonian_labels, parameters):
    filename = filename_spectrum_luttinger_parm(MminL, MmaxL, edge_states, N, window_of_lz, hamiltonian_labels,
                                                parameters)
    file1 = open(filename, 'rb')
    data = pickle.load(file1)
    file1.close()

    spectrum = data['spectrum']
    return spectrum


def filename_spectrum_lz_total_vals(MminL, MmaxL, edge_states, N, hamiltonian_labels, parameters):
    hamiltonian_args = [str(hamiltonian_labels[i]) + '_' + str(parameters[i]) for i in range(len(hamiltonian_labels))]
    args = [MminL, MmaxL, edge_states, N, 'Ham_lbls', *hamiltonian_args]
    args = [str(a) for a in args]
    filename = 'spectrum_lz_total_vals_' + '_'.join(args) + '.pkl'
    directory = directory_full_spectrum(MminL, MmaxL, edge_states, N)
    filename = directory + filename
    return filename


def directory_spectrum_eigenstates(MminL, MmaxL, edge_states, N):
    directory = 'pkl_data/eigenstates_full_spectrum_annulus/' + str(N) + '_particles/'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)

    sub_dir = '_'.join([str(MminL), str(MmaxL), str(edge_states)]) + '/'
    directory = directory + sub_dir
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def filename_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters):
    common_args = [MminL, MmaxL, edge_states, N, lz_val]
    common_args = [str(a) for a in common_args]
    filename = 'eigenstates_full_spectrum_' + '_'.join(common_args) + '_'

    hamiltonian_args = [str(hamiltonian_labels[i]) + '_' + str(parameters[i]) for i in range(len(hamiltonian_labels))]
    filename = filename + '_'.join(hamiltonian_args) + '.pkl'
    directory = directory_spectrum_eigenstates(MminL, MmaxL, edge_states, N)
    filename = directory + filename
    return filename


def read_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters):
    filename = filename_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters)
    file1 = open(filename, 'rb')
    dic = pickle.load(file1)
    file1.close()
    spectrum = dic['spectrum_eigenstates']
    return spectrum


def read_spectrum_eigenstates_from_file(filename):
    file1 = open(filename, 'rb')
    dic = pickle.load(file1)
    file1.close()
    spectrum = dic['spectrum_eigenstates']
    return spectrum


def write_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters, spectrum):
    filename = filename_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters)
    file1 = open(filename, 'wb')
    dic = {'spectrum_eigenstates': spectrum}
    pickle.dump(dic, file1)
    file1.close()
    print("wrote low lying spectrum into: " + filename)
    return 0


def write_spectrum_eigenstates_to_file(filename, spectrum):
    file1 = open(filename, 'wb')
    dic = {'spectrum_eigenstates': spectrum}
    pickle.dump(dic, file1)
    file1.close()
    return 0


def complete_matrices_directory(matrix_name, args):
    hilbert_space_string = "_".join([str(a) for a in args[:-1]])
    # directory = 'pkl_data/matrices/' + hilbert_space_string + '/'
    storage_dir = get_storage_dir()
    dir_N = storage_dir + 'pkl_data/matrices/' + str(args[3]) + '_particles/'
    if not os.path.exists(dir_N):
        os.mkdir(dir_N)
    dir_parms = dir_N + hilbert_space_string + '/'
    if not os.path.exists(dir_parms):
        os.mkdir(dir_parms)
    directory = dir_parms + matrix_name + '/'

    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def filename_matrix_pieces_directory(matrix_name, args):
    directory = complete_matrices_directory(matrix_name, args)

    args_str = [str(a) for a in args]
    one_string = "_".join(args_str)
    directory = directory + matrix_name + '_' + one_string + '/'

    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def filename_complete_matrix(matrix_name, args):
    args_str = [str(a) for a in args]
    one_string = "_".join(args_str)
    filename = matrix_name + '_' + one_string + '.npz'
    directory = complete_matrices_directory(matrix_name, args)
    filename = directory + filename
    return filename


def read_complete_matrix(matrix_name, args, output=1):
    # args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    filename = filename_complete_matrix(matrix_name, args)
    if EC.does_file_really_exist(filename):
        if not output:
            print("complete matrix exists")
            return 1
        npzfile = np.load(filename)
        row = npzfile['row']
        col = npzfile['col']
        matrix_elements = npzfile['matrix_elements']
        return row, col, matrix_elements

    print("complete matrix doesn't exist yet")
    return 0


def write_complete_matrix(matrix_name, args, row, col, matrix_elements):
    # args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    filename = filename_complete_matrix(matrix_name, args)
    np.savez(filename, col=col, row=row, matrix_elements=matrix_elements)
    return 0


def filename_matrix_piece(matrix_name, args, slice):
    directory = filename_matrix_pieces_directory(matrix_name, args)
    slice_str = str(slice[0]) + '_to_' + str(slice[1]) + '.npz'
    filename = directory + slice_str
    return filename


def read_matrix_piece(matrix_name, args, slice):
    filename = filename_matrix_piece(matrix_name, args, slice)
    npzfile = np.load(filename)
    col = npzfile['col']
    row = npzfile['row']
    matrix_elements = npzfile['matrix_elements']

    return row, col, matrix_elements


def write_matrix_piece(matrix_name, args, slice, row, col, matrix_elements):
    filename = filename_matrix_piece(matrix_name, args, slice)
    np.savez(filename, col=col, row=row, matrix_elements=matrix_elements)
    print("wrote matrix piece into file " + filename)
    return 0


def directory_parameter_files_annulus():
    directory = 'pkl_data/parameter_configuration_annulus/'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def filename_parameters_annulus(name):
    directory = directory_parameter_files_annulus()
    fullfilename = directory + name
    if fullfilename[-4:] != '.yml':
        fullfilename = fullfilename + '.yml'
    return fullfilename


def get_short_parameters_filename(full_filename):
    parameters_directory_path = directory_parameter_files_annulus()
    short_filename = full_filename[len(parameters_directory_path):]
    return short_filename


def filename_spectrum_vs_magnetic_flux(MminL, MmaxL, edge_states, N, hamiltonian_labels, parameters):
    hamiltonian_args = [str(hamiltonian_labels[i]) + '_' + str(parameters[i]) for i in range(len(hamiltonian_labels))]
    args = [MminL, MmaxL, edge_states, N, 'Ham_lbls', *hamiltonian_args]
    args = [str(a) for a in args]
    filename = 'spectrum_vs_magnetic_flux_' + "_".join(args) + '.pkl'
    directory = directory_full_spectrum(MminL, MmaxL, edge_states, N)
    filename = directory + filename
    return filename


def edge_subspace_matrices_directory(matrix_name, args):
    # args = [MminL,MmaxL,edge_states,N,lz_val,matrix_label]
    hilbert_space_string = "_".join([str(a) for a in args[:-1]])
    directory = 'pkl_data/matrices_edge_state_subspace/' + hilbert_space_string + '/'

    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def filename_edge_subspace_matrix(matrix_name, matrix_dim, args):
    args_str = [str(a) for a in args]
    one_string = "_".join(args_str)
    filename = matrix_name + '_matrix_dim=' + str(matrix_dim) + '_' + one_string + '.npz'
    directory = edge_subspace_matrices_directory(matrix_name, args)
    filename = directory + filename
    return filename


def read_edge_subspace_matrix(matrix_name, matrix_dim, args, output=1):
    # args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    filename = filename_edge_subspace_matrix(matrix_name, matrix_dim, args)
    if EC.does_file_really_exist(filename):
        if not output:
            print("complete matrix exists")
            return 1
        npzfile = np.load(filename)
        matrix = npzfile['matrix']
        return matrix

    print("complete matrix doesn't exist yet")
    return 0


def write_edge_subspace_matrix(matrix_name, args, matrix_dim, matrix):
    # args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    filename = filename_edge_subspace_matrix(matrix_name, matrix_dim, args)
    # filename = filename_edge_subspace_matrix(matrix_name, args)
    np.savez(filename, matrix=matrix)
    print("wrote subspace matrix into file " + filename)
    return 0


def directory_spectrum_subspace(MminL, MmaxL, edge_states, N):
    directory = 'pkl_data/spectrum_subspace_annulus/' + str(N) + '_particles/'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)

    sub_dir = '_'.join([str(MminL), str(MmaxL), str(edge_states)]) + '/'
    directory = directory + sub_dir
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def filename_lz_total_spectrum_low_lying_subspace(MminL, MmaxL, edge_states, N, hamiltonian_labels,
                                                  parameters, subspace_size):
    # not including interactions because we only use toy interactions for the calculation
    hamiltonian_args = [str(hamiltonian_labels[i]) + '_' + str(parameters[i]) for i in
                        range(1, len(hamiltonian_labels))]
    args = [MminL, MmaxL, edge_states, N, 'Ham_lbls', *hamiltonian_args, 'subspace_dim', subspace_size]
    args = [str(a) for a in args]
    filename = 'spectrum_lz_total_vals_' + '_'.join(args) + '.pkl'
    directory = directory_spectrum_subspace(MminL, MmaxL, edge_states, N)
    filename = directory + filename
    return filename


def filename_density_profile_groundstate(MminL, MmaxL, edge_states, N, hamiltonian_labels,
                                         parameters, num_points):
    directory = 'pkl_data/density_profiles/' + str(N) + '_particles/'
    storage_dir = get_storage_dir()
    directory = storage_dir + directory
    if not os.path.exists(directory):
        os.mkdir(directory)
    hamiltonian_args = [str(hamiltonian_labels[i]) + '_' + str(parameters[i]) for i in range(len(hamiltonian_labels))]
    args = [MminL, MmaxL, edge_states, N, num_points, 'Ham_lbls', *hamiltonian_args]
    args = [str(a) for a in args]
    filename = 'density_profile_groundstate_' + '_'.join(args)
    filename = directory + filename
    return filename


def read_spectrum_data_from_file(filename):
    file1 = open(filename, 'rb')
    data = pickle.load(file1)
    file1.close()

    spectrum = data['spectrum']
    return spectrum
