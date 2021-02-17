from DataManaging.ParametersAnnulus import *
from AnnulusFQH import BasisAndMBNonInteracting as GA, InteractionMatrices as LCA, MatricesAndSpectra as AMAS
from clusterScripts import scriptNames
from time import sleep
import networkx as nx
from datetime import datetime
from os import environ
import os, sys
import yaml
from DataManaging import fileManaging as FM
import itertools

if os.getcwd()[0] == "/":
    project_dir = os.path.dirname(__file__)
    project_dir = "/" + "/".join(project_dir.split("/")[1:-1])
    file = open(project_dir + '/configurations.yml', 'r')

else:
    file = open('../configurations.yml', 'r')
    project_dir = 'DUMMY'
docs = yaml.full_load(file)
file.close()
directories = docs['directories']
jobSenderDir = directories['job_sender_directory']

if os.getcwd()[0] == '/':
    sys.path.append(jobSenderDir)

import CondorJobSender


def limit_num_threads(num_threads=1):
    num_threads = str(num_threads)
    environ["OMP_NUM_THREADS"] = num_threads
    environ["OPENBLAS_NUM_THREADS"] = num_threads
    environ["MKL_NUM_THREADS"] = num_threads
    environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    environ["NUMEXPR_NUM_THREADS"] = num_threads
    return 0


def job_dict_create_basis(Mmin, Mmax, N, lz_val, requestMemory=None, request_cpus=None):
    job_dict = {}
    if lz_val == 'not_fixed':
        filename_basis = FM.filename_basis_annulus(Mmin, Mmax, N)
    else:
        filename_basis = FM.filename_basis_annulus_const_lz(Mmin, Mmax, N, lz_val)
    job_dict['batch_parameters'] = [{}]
    if FM.does_file_really_exist(filename_basis):
        print("basis already exists")
        job_dict['batch_parameters'] = []
    job_dict['py_script_path'] = project_dir + "/" + scriptNames.multiFile
    job_dict['kargs_dict'] = {}

    arguments = [Mmin, Mmax, N, lz_val, 'create_basis']
    arguments = [str(a) for a in arguments]
    arguments = " ".join(arguments)
    job_dict['kargs_dict']['Arguments'] = arguments

    if request_cpus:
        job_dict['kargs_dict']['request_cpus'] = request_cpus
    if requestMemory:
        job_dict['kargs_dict']['requestMemory'] = requestMemory

    return job_dict


def job_dict_create_two_particle_hamiltonian(Mmin, Mmax, potential_type, magnetic_flux=None, requestMemory=None,
                                             request_cpus=None):
    job_dict = {}
    job_dict['batch_parameters'] = [{}]
    job_dict['py_script_path'] = project_dir + "/" + scriptNames.multiFile
    job_dict['kargs_dict'] = {}

    arguments = [Mmin, Mmax, potential_type]
    if magnetic_flux != None:
        arguments = arguments + [magnetic_flux]
    arguments = arguments + ['create_two_particle_hamiltonian']
    arguments = [str(a) for a in arguments]
    arguments = " ".join(arguments)
    job_dict['kargs_dict']['Arguments'] = arguments

    if request_cpus:
        job_dict['kargs_dict']['request_cpus'] = request_cpus
    if requestMemory:
        job_dict['kargs_dict']['requestMemory'] = requestMemory

    return job_dict


def job_dict_create_matrix_pieces(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, speeding_parameter,
                                  requestMemory=None, request_cpus=None):
    job_dict = {}
    filename_args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states
    hilbert_space_size = GA.size_of_hilbert_space(Mmin, Mmax, N, lz_val)

    job_dict['speeding_parameter'] = speeding_parameter

    slice_size = int(hilbert_space_size / speeding_parameter)
    last_slice_size = hilbert_space_size - speeding_parameter * slice_size
    slice_start = 0
    slice_end = slice_size
    vars_matrix_slices = []
    for i in range(speeding_parameter):
        filename_martix_piece = FM.filename_matrix_piece(matrix_name, filename_args, [slice_start, slice_end - 1])
        if FM.does_file_really_exist(filename_martix_piece):
            print("matrix piece " + filename_martix_piece + " already exists!")
        else:
            vars_matrix_slices.append({'slice_start': slice_start, 'slice_end': slice_end})
        slice_start = slice_start + slice_size
        slice_end = slice_end + slice_size

    # taking care of last slice
    if last_slice_size > 0:
        slice_end = slice_start + last_slice_size
        filename_args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
        filename_martix_piece = FM.filename_matrix_piece(matrix_name, filename_args, [slice_start, slice_end - 1])
        if FM.does_file_really_exist(filename_martix_piece):
            print("matrix piece " + filename_martix_piece + " already exists!")
        else:
            vars_matrix_slices.append({'slice_start': slice_start, 'slice_end': slice_end})

    arguments = [matrix_name] + filename_args
    arguments = [str(a) for a in arguments]
    arguments = " ".join(arguments)
    arguments = arguments + " $(slice_start) $(slice_end)"
    job_dict['kargs_dict'] = {}

    job_dict['kargs_dict']['Arguments'] = arguments
    if request_cpus:
        job_dict['kargs_dict']['request_cpus'] = request_cpus
    if requestMemory:
        job_dict['kargs_dict']['requestMemory'] = requestMemory

    job_dict['batch_parameters'] = vars_matrix_slices
    job_dict['py_script_path'] = project_dir + "/" + scriptNames.piecesMBMatrix

    return job_dict


def job_dict_unite_matrix_pieces(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name,
                                 requestMemory=None, request_cpus=None):
    job_dict = {}
    filename_args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    job_dict['batch_parameters'] = [{}]
    job_dict['py_script_path'] = project_dir + "/" + scriptNames.uniteMatrixPieces
    arguments = [matrix_name] + filename_args
    arguments = [str(a) for a in arguments]
    arguments = " ".join(arguments)
    job_dict['kargs_dict'] = {}

    job_dict['kargs_dict']['Arguments'] = arguments

    if request_cpus:
        job_dict['kargs_dict']['request_cpus'] = request_cpus
    if requestMemory:
        job_dict['kargs_dict']['requestMemory'] = requestMemory

    return job_dict


def job_dict_get_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters, num_of_eigstates=20,
                          magnetic_flux=None, return_eigvecs=True, requestMemory=None, request_cpus=None):
    job_dict = {}

    job_dict['batch_parameters'] = [{}]
    job_dict['py_script_path'] = project_dir + "/" + scriptNames.multiFile
    job_dict['kargs_dict'] = {}

    if magnetic_flux != None:
        hamiltonian_labels = [hl + '_' + str(magnetic_flux) for hl in hamiltonian_labels]
    arguments = [MminL, MmaxL, edge_states, N, lz_val, *hamiltonian_labels, *parameters, num_of_eigstates,
                 return_eigvecs, 'calc_low_lying_spectrum']
    arguments = [str(a) for a in arguments]
    arguments = " ".join(arguments)
    job_dict['kargs_dict']['Arguments'] = arguments

    if request_cpus:
        job_dict['kargs_dict']['request_cpus'] = request_cpus
    if requestMemory:
        job_dict['kargs_dict']['requestMemory'] = requestMemory

    return job_dict


def job_dict_lz_resolved_spectrum(MminL, MmaxL, edge_states, N, lz_center_val, hamiltonian_labels, parameters,
                                  window_of_lz, num_of_eigstates,
                                  magnetic_flux=None, requestMemory=None, request_cpus=None):
    job_dict = {}

    job_dict['batch_parameters'] = [{}]
    job_dict['py_script_path'] = project_dir + "/" + scriptNames.multiFile
    job_dict['kargs_dict'] = {}

    if magnetic_flux != None:
        hamiltonian_labels = [hl + '_' + str(magnetic_flux) for hl in hamiltonian_labels]
    arguments = [MminL, MmaxL, edge_states, N, lz_center_val, *hamiltonian_labels,
                 *parameters, window_of_lz, num_of_eigstates, 'calc_lz_resolved_spectrum']

    arguments = [str(a) for a in arguments]
    arguments = " ".join(arguments)
    job_dict['kargs_dict']['Arguments'] = arguments

    if request_cpus:
        job_dict['kargs_dict']['request_cpus'] = request_cpus
    if requestMemory:
        job_dict['kargs_dict']['requestMemory'] = requestMemory

    return job_dict


def job_dict_complete_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, requestMemory=None,
                             request_cpus=None):
    job_dict = {}
    job_dict['batch_parameters'] = [{}]
    job_dict['py_script_path'] = project_dir + "/" + scriptNames.uniteMatrixPieces
    arguments = [MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, 'create_complete_matrix']
    arguments = [str(a) for a in arguments]
    arguments = " ".join(arguments)
    job_dict['kargs_dict'] = {}

    job_dict['kargs_dict']['Arguments'] = arguments

    if request_cpus:
        job_dict['kargs_dict']['request_cpus'] = request_cpus
    if requestMemory:
        job_dict['kargs_dict']['requestMemory'] = requestMemory
    return job_dict


def job_dict_luttinger_parameter(MminL, MmaxL, edge_states, N, hamiltonian_labels, parameters, num_of_eigstates,
                                 magnetic_flux=None, requestMemory=None, request_cpus=None):
    job_dict = {}

    job_dict['batch_parameters'] = [{}]
    job_dict['py_script_path'] = project_dir + "/" + scriptNames.multiFile
    job_dict['kargs_dict'] = {}

    if magnetic_flux != None:
        hamiltonian_labels = [hl + '_' + str(magnetic_flux) for hl in hamiltonian_labels]

    arguments = [MminL, MmaxL, edge_states, N, *hamiltonian_labels, *parameters, num_of_eigstates,
                 'calc_luttinger_parameter']

    arguments = [str(a) for a in arguments]
    arguments = " ".join(arguments)
    job_dict['kargs_dict']['Arguments'] = arguments

    if request_cpus:
        job_dict['kargs_dict']['request_cpus'] = request_cpus
    if requestMemory:
        job_dict['kargs_dict']['requestMemory'] = requestMemory

    return job_dict


def calc_lz_resolved_spectrum_DAG(dag_dir_name, jobs_comp_requirements, MminL, MmaxL, edge_states, N, lz_center_val,
                                  hamiltonian_labels,
                                  parameters, num_of_eigstates=20, magnetic_flux=None, window_of_lz='all'):
    if magnetic_flux != None:
        hamiltonian_labels = [hl + '_' + str(magnetic_flux) for hl in hamiltonian_labels]

    filename_full_spectrum = FM.filename_full_spectrum(MminL, MmaxL, edge_states, N, window_of_lz, hamiltonian_labels,
                                                       parameters)

    if FM.does_file_really_exist(filename_full_spectrum):
        print("spectrum already exists")
        return 0

    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states
    lz_total_vals = LCA.find_all_lz_total_values(Mmin, Mmax, N)

    if window_of_lz == 'all':
        lz_min = lz_total_vals[0]
        lz_max = lz_total_vals[-1]

    else:
        lz_min = max(lz_center_val - window_of_lz, lz_total_vals[0])
        lz_max = min(lz_center_val + window_of_lz, lz_total_vals[-1])

    jobs_information_dict = {}

    jobs_information_dict['calc_spectrum'] = job_dict_get_spectrum(MminL, MmaxL, edge_states, N, '$(lz_val)',
                                                                   hamiltonian_labels, parameters, num_of_eigstates,
                                                                   magnetic_flux, False,
                                                                   **jobs_comp_requirements['calc_spectrum'])
    spectrum_batch_parms = []
    for lz in range(lz_min, lz_max + 1):
        filename_spectrum = FM.filename_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz, hamiltonian_labels,
                                                           parameters)
        if not FM.does_file_really_exist(filename_spectrum):
            spectrum_batch_parms.append({'lz_val': str(lz)})

    jobs_information_dict['calc_spectrum']['batch_parameters'] = spectrum_batch_parms

    jobs_information_dict['unite_lz_resolved'] = job_dict_lz_resolved_spectrum(MminL, MmaxL, edge_states, N,
                                                                               lz_center_val, hamiltonian_labels,
                                                                               parameters, window_of_lz,
                                                                               num_of_eigstates, magnetic_flux,
                                                                               **jobs_comp_requirements[
                                                                                   'unite_lz_resolved'])
    dag_graph = nx.DiGraph()
    dag_graph.add_edges_from([('calc_spectrum', 'unite_lz_resolved')])
    send_minimal_dag_job(dag_graph, dag_dir_name, jobs_information_dict)
    return 0


def calc_luttinger_parameter_DAG(dag_dir_name, jobs_comp_requirements, MminL, MmaxL, edge_states, N, lz_center_val,
                                 hamiltonian_labels, parameters, num_of_eigstates=20, magnetic_flux=None):
    if magnetic_flux != None:
        hamiltonian_labels = [hl + '_' + str(magnetic_flux) for hl in hamiltonian_labels]
    window_of_lz = 'all'
    filename_full_spectrum = FM.filename_full_spectrum(MminL, MmaxL, edge_states, N, window_of_lz, hamiltonian_labels,
                                                       parameters)
    filename_lut_spectrum = FM.filename_spectrum_luttinger_parm(MminL, MmaxL, edge_states, N, window_of_lz,
                                                                hamiltonian_labels, parameters)

    if FM.does_file_really_exist(filename_full_spectrum) or FM.does_file_really_exist(filename_lut_spectrum):
        print("spectrum already exists. no need for cluster")
        return 0

    first_arc_lz = np.array([lz_center_val + i for i in range(N)])
    umbrella_lz = np.array([lz_center_val - N * edge_states + N * i for i in range(2 * edge_states + 1)])
    lz_vals_to_add_spectrum = np.concatenate((first_arc_lz, umbrella_lz))

    jobs_information_dict = {}

    jobs_information_dict['calc_spectrum'] = job_dict_get_spectrum(MminL, MmaxL, edge_states, N, '$(lz_val)',
                                                                   hamiltonian_labels, parameters, num_of_eigstates,
                                                                   magnetic_flux, False,
                                                                   **jobs_comp_requirements['calc_spectrum'])
    spectrum_batch_parms = []
    for lz in lz_vals_to_add_spectrum:
        filename_spectrum = FM.filename_low_lying_spectrum(MminL, MmaxL, edge_states, N, lz, hamiltonian_labels,
                                                           parameters)
        if not FM.does_file_really_exist(filename_spectrum):
            spectrum_batch_parms.append({'lz_val': str(lz)})

    jobs_information_dict['calc_spectrum']['batch_parameters'] = spectrum_batch_parms

    jobs_information_dict['luttinger_parameter'] = job_dict_lz_resolved_spectrum(MminL, MmaxL, edge_states, N,
                                                                                 lz_center_val, hamiltonian_labels,
                                                                                 parameters, window_of_lz,
                                                                                 num_of_eigstates, magnetic_flux,
                                                                                 **jobs_comp_requirements[
                                                                                     'luttinger_parameter'])
    dag_graph = nx.DiGraph()
    dag_graph.add_edges_from([('calc_spectrum', 'luttinger_parameter')])
    send_minimal_dag_job(dag_graph, dag_dir_name, jobs_information_dict)
    return 0


def get_spectrum_batch_jobs(batch_parameters, MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters,
                            num_of_eigstates=20, magnetic_flux=None, return_eigvecs=True, requestMemory=None,
                            request_cpus=None):
    kargs_dict = job_dict_get_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters,
                                       num_of_eigstates, magnetic_flux, return_eigvecs, requestMemory, request_cpus)[
        'kargs_dict']
    py_path = project_dir + "/" + scriptNames.multiFile
    CondorJobSender.send_batch_of_jobs_to_condor(py_path, 'get_spectrum', batch_parameters, **kargs_dict)
    return 0


def get_spectrum_job(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters, num_of_eigstates=20,
                     magnetic_flux=None, return_eigvecs=True, requestMemory=None, request_cpus=None):
    kargs_dict = job_dict_get_spectrum(MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels, parameters,
                                       num_of_eigstates, magnetic_flux, return_eigvecs, requestMemory, request_cpus)[
        'kargs_dict']
    py_path = project_dir + "/" + scriptNames.multiFile
    CondorJobSender.send_job_to_condor(py_path, 'get_spectrum', **kargs_dict)
    return 0


def get_spectrum_DAG(dag_dir_name, jobs_comp_requirements, MminL, MmaxL, edge_states, N, lz_val, hamiltonian_labels,
                     parameters, num_of_eigstates=20, magnetic_flux=None, speeding_parameter=1):
    hamiltonian_terms_names = ['interactions', 'confining_potential', 'SC_term', 'FM_term']

    if magnetic_flux != None:
        hamiltonian_labels = [hl + '_' + str(magnetic_flux) for hl in hamiltonian_labels]

    filename_spectrum_eigstates = FM.filename_spectrum_eigenstates(MminL, MmaxL, edge_states, N, lz_val,
                                                                   hamiltonian_labels, parameters)
    if FM.does_file_really_exist(filename_spectrum_eigstates):
        print("already created " + filename_spectrum_eigstates)
        return 0

    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states
    jobs_information_dict = {}
    dag_graph = nx.DiGraph()

    uniting_jobs = []
    # Create basis
    jobs_information_dict['create_basis'] = job_dict_create_basis(Mmin, Mmax, N, lz_val,
                                                                  **jobs_comp_requirements['create_basis'])

    # Create all unmade pieces
    for hlabel, matrix_name in zip(hamiltonian_labels, hamiltonian_terms_names):
        filename_matrix_args = [MminL, MmaxL, edge_states, N, lz_val, hlabel]
        filename_complete_matrix = FM.filename_complete_matrix(matrix_name, filename_matrix_args)
        if FM.does_file_really_exist(filename_complete_matrix):
            print("already created " + filename_complete_matrix)
        elif hlabel != 'None':
            jobname_pieces = 'create_pieces_' + matrix_name
            jobname_unite = 'unite_pieces_' + matrix_name
            uniting_jobs.append(jobname_unite)
            # Create all unmade pieces
            jobs_information_dict[jobname_pieces] = \
                job_dict_create_matrix_pieces(MminL, MmaxL, edge_states, N, lz_val, hlabel, matrix_name,
                                              speeding_parameter, **jobs_comp_requirements[jobname_pieces])

            # Unite all pieces
            jobs_information_dict[jobname_unite] = \
                job_dict_unite_matrix_pieces(MminL, MmaxL, edge_states, N, lz_val, hlabel, matrix_name,
                                             **jobs_comp_requirements[jobname_unite])

            dag_graph.add_edges_from([('create_basis', jobname_pieces), (jobname_pieces, jobname_unite)])

    jobs_information_dict['calc_spectrum'] = job_dict_get_spectrum(MminL, MmaxL, edge_states, N, lz_val,
                                                                   hamiltonian_labels, parameters, num_of_eigstates,
                                                                   magnetic_flux, True,
                                                                   **jobs_comp_requirements['calc_spectrum'])
    edges_from_unite_to_last_node = list(itertools.product(uniting_jobs, ['calc_spectrum']))
    dag_graph.add_edges_from(edges_from_unite_to_last_node)
    # print(list(dag_graph.nodes))

    send_minimal_dag_job(dag_graph, dag_dir_name, jobs_information_dict)
    return 0


def complete_MB_matrix_job(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, requestMemory=None,
                           request_cpus=None):
    filename_args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]

    filename_complete_matrix = FM.filename_complete_matrix(matrix_name, filename_args)
    if FM.does_file_really_exist(filename_complete_matrix):
        print("already created " + filename_complete_matrix)
        return 0
    kargs_dict = job_dict_complete_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name,
                                          requestMemory, request_cpus)['kargs_dict']
    py_path = project_dir + "/" + scriptNames.multiFile
    CondorJobSender.send_job_to_condor(py_path, 'create_complete_matrix', **kargs_dict)
    return 0


def complete_MB_matrix_DAG(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, dag_dir_name,
                           jobs_comp_requirements, speeding_parameter=1):
    filename_args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    Mmin = MminL - edge_states
    Mmax = MmaxL + edge_states

    filename_complete_matrix = FM.filename_complete_matrix(matrix_name, filename_args)
    if FM.does_file_really_exist(filename_complete_matrix):
        print("already created " + filename_complete_matrix)
        return 0

    jobs_information_dict = {}

    # Create basis
    jobs_information_dict['create_basis'] = job_dict_create_basis(Mmin, Mmax, N, lz_val,
                                                                  **jobs_comp_requirements['create_basis'])

    # Create all unmade pieces
    jobs_information_dict['create_matrix_pieces'] = \
        job_dict_create_matrix_pieces(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name,
                                      speeding_parameter, **jobs_comp_requirements['create_matrix_pieces'])

    # Unite all pieces
    jobs_information_dict['unite_matrix_pieces'] = \
        job_dict_unite_matrix_pieces(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name,
                                     **jobs_comp_requirements['unite_matrix_pieces'])

    # create and send DAG to cluster
    dag_graph = nx.DiGraph()
    dag_graph.add_edges_from(
        [('create_basis', 'create_matrix_pieces'), ('create_matrix_pieces', 'unite_matrix_pieces')])

    send_minimal_dag_job(dag_graph, dag_dir_name, jobs_information_dict)
    return 0


def send_minimal_dag_job(dag_graph, dag_dir_name, jobs_information_dict):
    dag_graph = remove_empty_nodes(dag_graph, jobs_information_dict)
    if len(dag_graph.nodes) == 1:
        jobname = list(dag_graph.nodes)[0]
        if len(jobs_information_dict[jobname]['batch_parameters']) == 1:
            CondorJobSender.send_job_to_condor(jobs_information_dict[jobname]['py_script_path'],
                                               jobname, **jobs_information_dict[jobname]['kargs_dict'])
        else:
            CondorJobSender.send_batch_of_jobs_to_condor(jobs_information_dict[jobname]['py_script_path'],
                                                         jobname, jobs_information_dict[jobname]['batch_parameters'],
                                                         **jobs_information_dict[jobname]['kargs_dict'])

    else:
        CondorJobSender.send_dag_job(dag_graph, dag_dir_name, jobs_information_dict)
    return 0


def remove_empty_nodes(dag, jobs_information_dict):
    job_names = list(dag.nodes)

    for jobname in job_names:
        if jobs_information_dict[jobname]['batch_parameters'] == []:
            parents = list(dag.predecessors(jobname))
            children = list(dag.successors(jobname))
            all_new_edges = list(itertools.product(parents, children))
            dag.add_edges_from(all_new_edges)
            dag.remove_node(jobname)

    return dag
