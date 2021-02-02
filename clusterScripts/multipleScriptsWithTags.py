import sys
import os
import yaml

file = open('../configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()

if os.getcwd()[0] == '/':
    working_dir = docs['directories']['working_dir']
    sys.path.append(working_dir)

from ATLASClusterInterface import JobSender as JS, errorCorrectionsAndTests as EC, MaSWrapperForATLAS as AMASW
from AnnulusFQH import InteractionMatrices as LCA
from AnnulusFQH import BasisAndMBNonInteracting as GA, MatricesAndSpectra as AMAS
from DataManaging import fileManaging as FM
from DataManaging.ParametersAnnulus import *
import numpy as np
from time import sleep

JS.limit_num_threads()
script_name = sys.argv[0]
m = 3
# Mmin = int(sys.argv[1])
# Mmax = int(sys.argv[2])
# MminLaughlin = int(sys.argv[3])
# MmaxLaughlin = int(sys.argv[4])
# N = int(sys.argv[5])
# potential_type = sys.argv[6]
# confining_potential_name = sys.argv[7]
# omega = float(sys.argv[8])
# R = float(sys.argv[9])
# theta = float(sys.argv[10])
# input_args = sys.argv
params_filename = sys.argv[1]
which_function = sys.argv[-1]

params = ParametersAnnulus(params_filename)

Mmin = params.MminLaughlin - params.edge_states
Mmax = params.MmaxLaughlin + params.edge_states


def create_two_particle_hamiltonian():
    if len(sys.argv) == 4:
        magnetic_flux = float(sys.argv[2])
        LCA.extract_two_particle_hamiltonian(Mmin, Mmax, params.potential_type, magnetic_flux)
    else:
        LCA.extract_two_particle_hamiltonian(Mmin, Mmax, params.potential_type)
    return 0


def create_all_basis():
    basis_list_all = GA.create_basis_annulus(Mmin, Mmax, params.N)

    if params.lz_val == 'not_fixed':

        print("do nothing for now")
        # lz_tot_vals = LCA.find_all_lz_total_values(Mmin, Mmax, N)
        # for lz_val in lz_tot_vals:
        #     try:
        #         basis_lz = FM.read_basis_annulus_const_lz(Mmin, Mmax, N, lz_val)
        #     except FileNotFoundError:
        #
        #         basis_ind = [np.where(vec == 1)[0] for vec in basis_list_all]
        #         lz_vals = np.array([sum(vec_ind) + N * Mmin for vec_ind in basis_ind])
        #         basis_lz = basis_list_all[lz_vals == lz_val]
        #         basis_lz = np.array(basis_lz, dtype=bool)
        #         FM.write_basis_annulus_const_lz(Mmin, Mmax, N, lz_val, basis_lz)
        #         FM.write_size_of_hilbert_space(Mmin, Mmax, N, lz_val, len(basis_lz))
    else:
        # lz_val = int(float(lz_val))
        try:
            basis_lz = FM.read_basis_annulus_const_lz(Mmin, Mmax, params.N, params.lz_val)
        except FileNotFoundError:

            basis_ind = [np.where(vec == 1)[0] for vec in basis_list_all]
            lz_vals = np.array([sum(vec_ind) + params.N * Mmin for vec_ind in basis_ind])
            basis_lz = basis_list_all[lz_vals == params.lz_val]
            basis_lz = np.array(basis_lz, dtype=bool)

            FM.write_basis_annulus_const_lz(Mmin, Mmax, params.N, params.lz_val, basis_lz)
            FM.write_size_of_hilbert_space(Mmin, Mmax, params.N, params.lz_val, len(basis_lz))

    return 0


def create_relevant_basis_list():
    """
    doesn't work with external set of lz_val
    If necessary, will add later
    :return:
    """
    if params.basis_list_queue == 'P' or params.basis_list_queue == 'M':
        JS.limit_num_threads(4)
    if params.lz_val == 'not_fixed':
        basis_list = GA.create_basis_annulus(Mmin, Mmax, params.N)
    else:
        basis_list = GA.create_basis_annulus_const_lz(Mmin, Mmax, params.N, params.lz_val)
    return 0


def calc_hamiltonian_pieces():
    if params.matrix_pieces_queue == 'P' or params.matrix_pieces_queue == 'M':
        JS.limit_num_threads(4)

    if len(sys.argv) == 4:
        lz_val = sys.argv[2]
        if lz_val != 'not_fixed':
            lz_val = int(float(lz_val))
        AMAS.calc_hamiltonian_pieces(params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N,
                                     lz_val, params.hamiltonian_labels, params_filename)
    else:
        AMAS.calc_hamiltonian_pieces(params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N,
                                     params.lz_val, params.hamiltonian_labels, params_filename)
    return 0


def setting_external_lz_val_h_parameters(pull_from_params=1, is_magnetic_index=0):
    lz_val = None
    parameters = None

    num_arguments = len(sys.argv)
    if is_magnetic_index:
        num_arguments = num_arguments - 1

    if num_arguments == 4:
        lz_val = sys.argv[2 + is_magnetic_index]

    if num_arguments == 7:
        parameters = sys.argv[2 + is_magnetic_index:6 + is_magnetic_index]
        parameters = [float(p) for p in parameters]

    if num_arguments == 8:
        lz_val = sys.argv[2 + is_magnetic_index]
        parameters = sys.argv[3 + is_magnetic_index:7 + is_magnetic_index]
        parameters = [float(p) for p in parameters]

    if pull_from_params:
        if not lz_val:
            lz_val = params.lz_val
        if not parameters:
            parameters = params.h_parameters

    return lz_val, parameters


def calc_low_lying_spectrum():
    if params.spectrum_eigenstates_queue == 'P' or params.spectrum_eigenstates_queue == 'M':
        JS.limit_num_threads(4)

    lz_val, parameters = setting_external_lz_val_h_parameters()

    AMAS.get_low_lying_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N,
                                lz_val, params.hamiltonian_labels, parameters, params_filename)


def calc_spectrum_eigenstates_with_flux():
    if params.spectrum_eigenstates_queue == 'P' or params.spectrum_eigenstates_queue == 'M':
        JS.limit_num_threads(4)

    if params.spectrum_job_manager_queue == 'P' or params.spectrum_eigenstates_queue == 'M':
        JS.limit_num_threads(4)

    magnetic_flux_index = int(sys.argv[2])
    params.setHamiltonianLabels(magnetic_flux_index)
    AMAS.get_low_lying_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N,
                                params.lz_val, params.hamiltonian_labels, params.h_parameters, params_filename)
    return


def spectrum_eigenstates_manager():
    if params.managing_job_queue == 'P' or params.managing_job_queue == 'M':
        JS.limit_num_threads(4)

    if params.is_hilbert_space_small:
        if params.is_magnetic_flux:
            magnetic_flux_index = int(sys.argv[2])
            params.setHamiltonianLabels(magnetic_flux_index)
        lz_val, parameters = setting_external_lz_val_h_parameters(1, params.is_magnetic_flux)
        AMAS.get_low_lying_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N,
                                    lz_val, params.hamiltonian_labels, parameters, params_filename, 0)
        return 0

    magnetic_flux_index = 'None'
    if params.is_magnetic_flux:
        magnetic_flux_index = int(sys.argv[2])
    lz_val, parameters = setting_external_lz_val_h_parameters(0, params.is_magnetic_flux)

    AMASW.calc_low_lying_spectrum_from_scratch(params_filename, magnetic_flux_index)
    return 0


def spectrum_eigenstates_batch():
    if params.spectrum_eigenstates_queue == 'P' or params.spectrum_eigenstates_queue == 'M':
        JS.limit_num_threads(4)

    min_magnetic_flux_index = int(sys.argv[2])
    max_magnetic_flux_index = int(sys.argv[3])

    for magnetic_flux_index in range(min_magnetic_flux_index, max_magnetic_flux_index + 1):
        params.setHamiltonianLabels(magnetic_flux_index)
        AMAS.get_low_lying_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N,
                                    params.lz_val, params.hamiltonian_labels, params.h_parameters, params_filename, 0)
    return 0


def calc_full_spectrum():
    lz_val, parameters = setting_external_lz_val_h_parameters(0)
    AMASW.calc_full_low_lying_spectrum(params_filename, lz_val, parameters)
    return 0


def calc_lz_vals_for_spectrum_manager():
    if params.is_magnetic_flux:
        params.setHamiltonianLabels(0)
    if params.is_hilbert_space_small:
        lz_val, parameters = setting_external_lz_val_h_parameters()
        AMAS.calc_lz_total_for_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N,
                                        lz_val, params.hamiltonian_labels, parameters, params_filename, 0)
        return 0

    lz_val, parameters = setting_external_lz_val_h_parameters(0)
    AMASW.calc_lz_vals_for_spectrum_from_scratch(params_filename, lz_val, parameters)

    return 0


def calc_lz_vals_for_spectrum():
    lz_val, parameters = setting_external_lz_val_h_parameters()

    AMAS.calc_lz_total_for_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N,
                                    lz_val, params.hamiltonian_labels, parameters, params_filename, 0)
    return 0


def calc_luttinger_parameter():
    if params.luttinger_parm_queue == 'P' or params.luttinger_parm_queue == 'M':
        JS.limit_num_threads(4)

    AMAS.calc_luttinger_parameter_from_scratch(params_filename, params.lz_laughlin, False)


def calc_density_profile():
    AMAS.create_density_profile(params_filename)


if which_function == 'two_particle_hamiltonian':
    create_two_particle_hamiltonian()
if which_function == 'create_all_basis_lists':
    # create_all_basis()
    create_relevant_basis_list()
# if which_function == '2_point_function_graph_data':
#     create_2_point_function_graph_data()
# if which_function == '2_point_function_values':
#     create_2_point_function_values()
if which_function == 'calc_hamiltonian_pieces':
    calc_hamiltonian_pieces()
if which_function == 'calc_low_spectrum_eigenstates':
    calc_low_lying_spectrum()
if which_function == 'spectrum_eigenstates_manager':
    spectrum_eigenstates_manager()
if which_function == 'spectrum_eigenstates_batch':
    spectrum_eigenstates_batch()
if which_function == 'spectrum_lz_vals_manager':
    calc_lz_vals_for_spectrum_manager()
if which_function == 'spectrum_lz_calc':
    calc_lz_vals_for_spectrum()
if which_function == 'calc_full_spectrum':
    calc_full_spectrum()
if which_function == 'calc_luttinger_parameter':
    calc_luttinger_parameter()
if which_function == 'calc_density_profile':
    calc_density_profile()
if which_function == 'calc_spectrum_eigenstates_with_flux':
    calc_spectrum_eigenstates_with_flux()
