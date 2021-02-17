import sys
import os
import yaml

if os.getcwd()[0] == "/":
    project_dir = os.path.dirname(__file__)
    project_dir = "/" + "/".join(project_dir.split("/")[1:-1])
    file = open(project_dir + '/configurations.yml', 'r')

else:
    file = open('../configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()

if os.getcwd()[0] == '/':
    working_dir = docs['directories']['working_dir']
    sys.path.append(working_dir)

# from ATLASClusterInterface import JobSender as JS, errorCorrectionsAndTests as EC, MaSWrapperForATLAS as AMASW
from ATLASClusterInterface import JobSenderFQH as JS
from AnnulusFQH import InteractionMatrices as LCA
from AnnulusFQH import BasisAndMBNonInteracting as GA, MatricesAndSpectra as AMAS
from DataManaging import fileManaging as FM
# from DataManaging.ParametersAnnulus import *
import numpy as np

JS.limit_num_threads()
script_name = sys.argv[0]
which_function = sys.argv[-1]


def create_basis():
    Mmin, Mmax, N, lz_val = sys.argv[1:-1]
    Mmin = int(Mmin)
    Mmax = int(Mmax)
    N = int(N)
    if lz_val == 'not_fixed':
        GA.create_basis_annulus(Mmin, Mmax, N)
    else:
        lz_val = int(lz_val)
        GA.create_basis_annulus_const_lz(Mmin, Mmax, N, lz_val)
    return 0


def create_two_particle_hamiltonian():
    Mmin, Mmax, potential_type = sys.argv[1:4]
    Mmin = int(Mmin)
    Mmax = int(Mmax)
    if len(sys.argv) == 5:
        magnetic_flux = float(sys.argv[4])
        LCA.extract_two_particle_hamiltonian(Mmin, Mmax, potential_type, magnetic_flux)
    else:
        LCA.extract_two_particle_hamiltonian(Mmin, Mmax, potential_type)
    return 0


def calc_low_lying_spectrum():
    MminL, MmaxL, edge_states, N, lz_val = sys.argv[1:6]
    MminL = int(MminL)
    MmaxL = int(MmaxL)
    edge_states = int(edge_states)
    N = int(N)
    if lz_val != 'not_fixed':
        lz_val = int(lz_val)
    hamiltonian_labels = sys.argv[6:10]
    parameters = sys.argv[10:14]
    parameters = [float(p) for p in parameters]
    num_of_eigstates = int(sys.argv[14])
    return_eigstates = True
    if len(sys.argv) == 17 and sys.argv[15] == 'False':
        return_eigstates = False

    AMAS.get_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters, num_of_eigstates,
                                0, return_eigstates)
    return 0


def create_complete_matrix():
    MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name = sys.argv[1:8]
    MminL = int(MminL)
    MmaxL = int(MmaxL)
    edge_states = int(edge_states)
    N = int(N)
    if lz_val != 'not_fixed':
        lz_val = int(lz_val)
    AMAS.create_complete_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name)
    return 0


def calc_lz_resolved_spectrum():
    MminL, MmaxL, edge_states, N, lz_center_val = sys.argv[1:6]
    MminL = int(MminL)
    MmaxL = int(MmaxL)
    edge_states = int(edge_states)
    N = int(N)
    lz_center_val = int(lz_center_val)
    hamiltonian_labels = sys.argv[6:10]
    parameters = sys.argv[10:14]
    parameters = [float(p) for p in parameters]
    window_of_lz = sys.argv[14]
    if not window_of_lz == 'all':
        window_of_lz = int(window_of_lz)
    num_of_eigstates = int(sys.argv[15])

    AMAS.calc_lz_resolved_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz_center_val, hamiltonian_labels,
                                             parameters, window_of_lz, num_of_eigstates)
    return 0


def calc_luttinger_parameter():
    MminL, MmaxL, edge_states, N = sys.argv[1:5]
    MminL = int(MminL)
    MmaxL = int(MmaxL)
    edge_states = int(edge_states)
    N = int(N)
    hamiltonian_labels = sys.argv[5:9]
    parameters = sys.argv[9:13]
    parameters = [float(p) for p in parameters]
    num_of_eigstates = int(sys.argv[13])

    AMAS.calc_luttinger_parameter(MminL, MmaxL, edge_states, N, hamiltonian_labels, parameters, num_of_eigstates)
    return 0


def calc_spectra_for_range_lz():
    MminL, MmaxL, edge_states, N, lz_min, lz_max = sys.argv[1:7]
    MminL = int(MminL)
    MmaxL = int(MmaxL)
    edge_states = int(edge_states)
    N = int(N)
    lz_min = int(lz_min)
    lz_max = int(lz_max)
    hamiltonian_labels = sys.argv[7:11]
    parameters = sys.argv[11:15]
    parameters = [float(p) for p in parameters]
    num_of_eigstates = int(sys.argv[15])
    return_eigvectors = sys.argv[16]
    if return_eigvectors == 'True':
        return_eigvectors = True
    else:
        return_eigvectors = False

    for lz in range(lz_min, lz_max + 1):
        if return_eigvectors:
            filename_spectrum = FM.filename_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz, hamiltonian_labels,
                                                                 parameters)
        else:
            filename_spectrum = FM.filename_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz, hamiltonian_labels,
                                                               parameters)
        if not FM.does_file_really_exist(filename_spectrum):
            AMAS.get_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz, hamiltonian_labels, parameters,
                                        num_of_eigstates, 0, return_eigvectors)
    return 0


if which_function == 'create_basis':
    create_basis()
if which_function == 'create_two_particle_hamiltonian':
    create_two_particle_hamiltonian()
if which_function == 'calc_low_lying_spectrum':
    calc_low_lying_spectrum()
if which_function == 'create_complete_matrix':
    create_complete_matrix()
if which_function == 'calc_lz_resolved_spectrum':
    calc_lz_resolved_spectrum()
if which_function == 'calc_spectra_for_range_lz':
    calc_spectra_for_range_lz()
if which_function == 'calc_luttinger_parameter':
    calc_luttinger_parameter()


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
