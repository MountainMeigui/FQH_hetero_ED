from time import sleep

from AnnulusFQH import BasisAndMBNonInteracting as GA, InteractionMatrices as LCA, MatricesAndSpectra as AMAS
from DataManaging import fileManaging as FM, graphData
from DataManaging.ParametersAnnulus import *
from ATLASClusterInterface import JobSender as JS, errorCorrectionsAndTests as EC
from clusterScripts import scriptNames
import os


def number_of_files_needed_in_matrix_pieces_directory(hilbert_space_size, speeding_parameter):
    if hilbert_space_size % speeding_parameter == 0:
        number_of_pieces = speeding_parameter
    else:
        number_of_pieces = speeding_parameter + 1
    return number_of_pieces


def all_pieces_present_in_matrix_pieces_directory(matrix_name, MminL, MmaxL, edge_states, N, lz_val, matrix_label,
                                                  hilbert_space_dim, speeding_parameter):
    needed_number_of_pieces = number_of_files_needed_in_matrix_pieces_directory(hilbert_space_dim, speeding_parameter)
    args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    filename_directory = FM.filename_matrix_pieces_directory(matrix_name, args)
    files_in_directory = os.listdir(filename_directory)
    filename_directory = [f for f in files_in_directory if f[-4:] == '.npz']
    num_files_in_directory = len(files_in_directory)
    if num_files_in_directory < needed_number_of_pieces:
        return False
    return True


def unite_and_write_full_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, params_filename):
    params = ParametersAnnulus(params_filename)
    queue = params.write_unite_matrix_queue
    mem, vmem = JS.get_mem_vmem_vals(queue, params.write_unite_matrix_mem, params.write_unite_matrix_vmem)
    args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    filename_complete_matrix = FM.filename_complete_matrix(matrix_name, args)
    if EC.does_file_really_exist(filename_complete_matrix):
        print("matrix already written")
        return 0

    args = [matrix_name, MminL, MmaxL, edge_states, N, lz_val, matrix_label]
    filename = 'writing_matrix_' + "_".join([str(a) for a in args])
    filename = EC.make_job_name_short_again(filename)
    JS.send_job(scriptNames.uniteMatrixPieces, queue=queue, mem=mem, vmem=vmem, pbs_filename=filename,
                script_args=args)
    return 0


def create_two_particle_hamiltonian(params_filename, magnetic_flux_index=None):
    params = ParametersAnnulus(params_filename)
    queue = params.two_particle_queue
    mem = params.two_particle_mem
    vmem = params.two_particle_vmem
    mem, vmem = JS.get_mem_vmem_vals(queue, mem, vmem)
    Mmin = params.MminLaughlin - params.edge_states
    Mmax = params.MmaxLaughlin + params.edge_states
    if magnetic_flux_index != None:
        magnetic_flux = params.flux_range[magnetic_flux_index]
        two_particle_ham_filename = FM.filename_two_particle_matrices(Mmin, Mmax,
                                                                      params.potential_type + '_' + str(magnetic_flux))
    else:
        two_particle_ham_filename = FM.filename_two_particle_matrices(Mmin, Mmax, params.potential_type)

    if EC.does_file_really_exist(two_particle_ham_filename):
        print("2-particle hamiltonian already exists")

    else:
        print("creating 2-particle hamiltonian")
        if magnetic_flux_index != None:
            args = [params.params_full_filename, magnetic_flux, 'two_particle_hamiltonian']
        else:
            args = [params.params_full_filename, 'two_particle_hamiltonian']
        pbs_filename = '2_particle_ham_Mmin=' + str(Mmin) + '_Mmax=' + str(
            Mmax) + '_potential_type=' + params.potential_type
        pbs_filename = EC.make_job_name_short_again(pbs_filename)
        JS.send_job(scriptNames.multiFile, pbs_filename=pbs_filename, script_args=args,
                    queue=queue, mem=mem, vmem=vmem)

        two_particle_ham_filename = FM.filename_two_particle_matrices(Mmin, Mmax, params.potential_type)
        while not EC.does_file_really_exist(two_particle_ham_filename):
            sleep(20)
        print("done writing two-particle hamiltonian")


def create_basis_lists(params_filename, lz_val_external=None):
    params = ParametersAnnulus(params_filename)
    queue = params.basis_list_queue
    mem, vmem = JS.get_mem_vmem_vals(params.basis_list_queue, params.basis_list_mem, params.basis_list_vmem)

    Mmin = params.MminLaughlin - params.edge_states
    Mmax = params.MmaxLaughlin + params.edge_states
    if lz_val_external:
        lz_val = lz_val_external
    else:
        lz_val = params.lz_val
    if lz_val == 'not_fixed':
        # basis_filename = FM.filename_basis_annulus(Mmin, Mmax, params.N)
        basis_filename = FM.filename_basis_annulus(0, Mmax - Mmin, params.N)
    else:
        basis_filename = FM.filename_basis_annulus_const_lz(0, Mmax - Mmin, params.N, lz_val - params.N * Mmin)
    if not EC.does_file_really_exist(basis_filename):
        print("writing basis lists")
        if lz_val_external:
            args = [params.params_full_filename, lz_val, 'create_all_basis_lists']
        else:
            args = [params.params_full_filename, 'create_all_basis_lists']
        # filename = 'create_basis_' + params.params_filename_no_path
        filename = 'create_basis_Mmin=' + str(Mmin) + '_Mmax=' + str(Mmax) + '_N=' + params.N
        JS.send_job(scriptNames.multiFile, pbs_filename=filename, script_args=args,
                    queue=queue, mem=mem, vmem=vmem)

        while not EC.does_file_really_exist(basis_filename):
            sleep(60)
        print("done writing basis")
    else:
        print("basis already written")
    return 0


def calc_low_lying_spectrum_from_scratch(params_filename, magnetic_flux_index='None', send_jobs=True, wait=True):
    params = ParametersAnnulus(params_filename)
    if magnetic_flux_index != 'None':
        params.setHamiltonianLabels(magnetic_flux_index)
    create_basis_lists(params.params_full_filename)
    create_two_particle_hamiltonian(params_filename, magnetic_flux_index)

    all_matrices_filenames = AMAS.calc_hamiltonian_pieces(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                          params.N, params.lz_val, params.hamiltonian_labels,
                                                          params_filename, send_jobs)

    if not wait and not EC.all_files_exist(all_matrices_filenames):
        print("not all matrices written yet, and I'm not a patient woman")
        return 0

    while not EC.all_files_exist(all_matrices_filenames):
        sleep(120)

    # actual calculation of low lying spectrum
    filename_spectrum = FM.filename_spectrum_eigenstates(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                         params.N, params.lz_val, params.hamiltonian_labels,
                                                         params.h_parameters)
    if EC.does_file_really_exist(filename_spectrum):
        print("we did this spectrum calculation already")
        return filename_spectrum

    queue = params.spectrum_eigenstates_queue
    mem, vmem = JS.get_mem_vmem_vals(queue, params.spectrum_eigenstates_mem, params.spectrum_eigenstates_vmem)
    args = [params_filename, magnetic_flux_index]
    args = args + ['calc_spectrum_eigenstates_with_flux']
    # jobname = 'spectrum_eigenstates_' + params.params_filename_no_path
    jobname = 'spectrum_eigenstates_magnetic_flux_index=' + str(magnetic_flux_index)
    jobname = EC.make_job_name_short_again(jobname)
    JS.send_job(scriptNames.multiFile, pbs_filename=jobname, script_args=args,
                queue=queue, mem=mem, vmem=vmem, ppn=params.ppn)

    return filename_spectrum


def calc_lz_vals_for_spectrum_from_scratch(params_filename, lz_val_external=None, parameters_external=None):
    params = ParametersAnnulus(params_filename)
    if lz_val_external:
        lz_val = lz_val_external
    else:
        lz_val = params.lz_val

    if parameters_external:
        parameters = parameters_external
    else:
        parameters = params.h_parameters

    filename_lz_spectrum = FM.filename_spectrum_lz_total_vals(params.MminLaughlin, params.MmaxLaughlin,
                                                              params.edge_states, params.N, params.hamiltonian_labels,
                                                              parameters)
    if EC.does_file_really_exist(filename_lz_spectrum):
        print("already created " + filename_lz_spectrum)
        return filename_lz_spectrum

    filename_spectrum = calc_low_lying_spectrum_from_scratch(params_filename, lz_val_external, parameters_external)
    while not EC.does_file_really_exist(filename_spectrum):
        sleep(60)
    matrix_name = 'total_angular_momentum'
    matrix_label = params.total_ang_momentum_label
    args = [params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N, lz_val, matrix_label]
    AMAS.create_matrix_pieces(params.MminLaughlin, params.MmaxLaughlin, params.edge_states, params.N, lz_val,
                              matrix_label, matrix_name, params_filename)

    filename_lz_matrix = FM.filename_complete_matrix(matrix_name, args)
    while not EC.does_file_really_exist(filename_lz_matrix):
        sleep(60)

    queue = params.lz_spectrum_queue
    mem, vmem = JS.get_mem_vmem_vals(queue, params.lz_spectrum_mem, params.lz_spectrum_vmem)
    args = [params_filename]
    if lz_val_external:
        args = args + [lz_val_external]
    if parameters_external:
        args = args + parameters_external
    args = args + ['spectrum_lz_calc']
    jobname = 'calc_lz_spectrum_' + '_'.join([str(a) for a in args])
    jobname = EC.make_job_name_short_again(jobname)
    JS.send_job(scriptNames.multiFile, pbs_filename=jobname, script_args=args,
                queue=queue, mem=mem, vmem=vmem)

    return filename_lz_spectrum


def calc_lz_spectrum_wrap_job(params_filename, lz_val_external=None, parameters_external=None, magnetic_flux_index=0):
    params = ParametersAnnulus(params_filename)
    if lz_val_external:
        lz_val = lz_val_external
    else:
        lz_val = params.lz_val

    if parameters_external:
        parameters = parameters_external
    else:
        parameters = params.h_parameters

    if params.is_magnetic_flux:
        params.setHamiltonianLabels(magnetic_flux_index)
    filename_lz_spectrum = FM.filename_spectrum_lz_total_vals(params.MminLaughlin, params.MmaxLaughlin,
                                                              params.edge_states, params.N, params.hamiltonian_labels,
                                                              parameters)
    if EC.does_file_really_exist(filename_lz_spectrum):
        print("already created " + filename_lz_spectrum)
        return 0
    queue = params.lz_spectrum_manager_queue
    mem, vmem = JS.get_mem_vmem_vals(queue, params.lz_spectrum_manager_mem, params.lz_spectrum_manager_vmem)
    args = [params_filename]
    if lz_val_external:
        args = args + [lz_val_external]
    if parameters_external:
        args = args + parameters_external
    args = args + ['spectrum_lz_vals_manager']
    jobname = 'lz_spectrum_manager_params_file' + params.params_filename_no_path
    jobname = EC.make_job_name_short_again(jobname)
    JS.send_job(scriptNames.multiFile, pbs_filename=jobname, script_args=args,
                queue=queue, mem=mem, vmem=vmem)
    return 0


def calc_full_low_lying_spectrum(params_filename, send_jobs=True, wait=False, lz_val_middle=None,
                                 parameters_external=None, window_of_lz='all'):
    params = ParametersAnnulus(params_filename)
    if not lz_val_middle:
        lz_val = params.lz_laughlin
    else:
        lz_val = lz_val_middle
    if parameters_external:
        parameters = parameters_external
    else:
        parameters = params.h_parameters

    filename_full_spectrum = FM.filename_full_spectrum(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                       params.N, window_of_lz, params.hamiltonian_labels,
                                                       parameters)
    if EC.does_file_really_exist(filename_full_spectrum):
        full_spectrum = graphData.plot_spectrum_graph_data_from_file(filename_full_spectrum)
        return full_spectrum

    queue = params.spectrum_job_manager_queue
    mem, vmem = JS.get_mem_vmem_vals(queue, params.spectrum_job_manager_mem, params.spectrum_job_manager_vmem)
    Mmin = params.MminLaughlin - params.edge_states
    Mmax = params.MmaxLaughlin + params.edge_states

    lz_total_vals = LCA.find_all_lz_total_values(Mmin, Mmax, params.N)
    if window_of_lz == 'all':
        lz_min = lz_total_vals[0]
        lz_max = lz_total_vals[-1]

    else:
        lz_min = max(lz_val - window_of_lz, lz_total_vals[0])
        lz_max = min(lz_val + window_of_lz, lz_total_vals[-1])

    all_file_names = []
    for lz in range(lz_min, lz_max + 1):
        filename_lz_spectrum = FM.filename_spectrum_eigenstates(params.MminLaughlin, params.MmaxLaughlin,
                                                                params.edge_states, params.N, lz,
                                                                params.hamiltonian_labels, parameters)
        if EC.does_file_really_exist(filename_lz_spectrum):
            spectrum = FM.read_spectrum_eigenstates(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                    params.N, lz, params.hamiltonian_labels,
                                                    parameters)
            size_hilbert_space = GA.size_of_hilbert_space(Mmin, Mmax, params.N, lz_val)
            if len(spectrum) >= params.num_of_eigstates or size_hilbert_space <= params.num_of_eigstates:
                print("we did this lz spectrum calculation already!")
            else:
                if send_jobs:
                    sleep(2)
                    print("finding the spectrum")
                    args = [params_filename, lz]
                    if parameters_external:
                        args = args + parameters_external
                    args = args + ['spectrum_eigenstates_manager']
                    # job_name = 'spectrum_job_manager_' + params.params_filename_no_path
                    job_name = 'full_spectrum_N=' + str(params.N) + '_MminL=' + str(
                        params.MminLaughlin) + '_MmaxL=' + str(
                        params.MmaxLaughlin) + '_edges=' + str(params.edge_states)
                    job_name = EC.make_job_name_short_again(job_name)
                    JS.send_job(scriptNames.multiFile, pbs_filename=job_name, script_args=args,
                                queue=queue, mem=mem, vmem=vmem)

                all_file_names.append(filename_lz_spectrum)


        else:
            if send_jobs:
                sleep(2)
                print("finding the spectrum")
                args = [params_filename, lz]
                if parameters_external:
                    args = args + parameters_external
                args = args + ['spectrum_eigenstates_manager']
                # job_name = 'spectrum_job_manager_' + params.params_filename_no_path
                job_name = 'full_spectrum_N=' + str(params.N) + '_MminL=' + str(params.MminLaughlin) + '_MmaxL=' + str(
                    params.MmaxLaughlin) + '_edges=' + str(params.edge_states)
                job_name = EC.make_job_name_short_again(job_name)
                JS.send_job(scriptNames.multiFile, pbs_filename=job_name, script_args=args,
                            queue=queue, mem=mem, vmem=vmem)

            all_file_names.append(filename_lz_spectrum)

    while not EC.all_files_exist(all_file_names):
        if not wait:
            print("not done creating spectrum!")
            return
        sleep(100)

    full_spectrum = {}
    print(range(lz_min, lz_max + 1))
    for lz in range(lz_min, lz_max + 1):
        spec_states = FM.read_spectrum_eigenstates(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                   params.N, lz, params.hamiltonian_labels, parameters)
        spectrum = np.array([spec_states[i][0] for i in range(len(spec_states))])
        print("***************")
        print(spectrum)
        full_spectrum[lz] = spectrum

    title = 'low lying spectrum of system with N=' + str(params.N) + ' MminL=' + str(
        params.MminLaughlin) + ' MmaxL=' + str(params.MmaxLaughlin) + '\nham_lbls: ' + ' '.join(
        params.hamiltonian_labels) + '\nLz laughlin at ' + str(params.lz_laughlin)
    xlabel = 'Lz total'
    ylabel = 'Energy'
    graphData.write_spectrum_data_to_file(filename_full_spectrum, full_spectrum, title, xlabel, ylabel)

    return full_spectrum


def send_job_full_spectrum(params_filename, lz_val_middle=None, parameters_external=None):
    params = ParametersAnnulus(params_filename)

    filename_spectrum_eigenstates = FM.filename_spectrum_eigenstates(params.MminLaughlin, params.MmaxLaughlin,
                                                                     params.edge_states, params.N, params.lz_val,
                                                                     params.hamiltonian_labels, params.h_parameters)
    if EC.does_file_really_exist(filename_spectrum_eigenstates):
        print("file " + filename_spectrum_eigenstates + " already created")
        return 0

    queue = params.managing_job_queue
    mem, vmem = JS.get_mem_vmem_vals(queue, params.managing_job_mem, params.managing_job_vmem)
    args = [params_filename]
    if lz_val_middle:
        args = args + [lz_val_middle]
    if parameters_external:
        args = args + parameters_external
    # args = args + ['calc_full_spectrum']
    # calculates low lying spectrum
    args = args + ['spectrum_eigenstates_manager']
    jobname = 'full_spectrum_' + params.params_filename_no_path
    jobname = EC.make_job_name_short_again(jobname)
    JS.send_job(scriptNames.multiFile, pbs_filename=jobname, script_args=args,
                queue=queue, mem=mem, vmem=vmem)
    return 0


def calc_full_spectrum_magnetic_flux_range(params_filename, wait=True, send_jobs=True, batch_size=None):
    params = ParametersAnnulus(params_filename)
    flux_range_size = len(params.flux_range)

    params.setHamiltonianLabels()
    filename_spectrum_vs_flux_range = FM.filename_spectrum_vs_magnetic_flux(params.MminLaughlin, params.MmaxLaughlin,
                                                                            params.edge_states, params.N,
                                                                            params.hamiltonian_labels,
                                                                            params.h_parameters)
    if EC.does_file_really_exist(filename_spectrum_vs_flux_range):
        spectrum = graphData.plot_spectrum_graph_data_from_file(filename_spectrum_vs_flux_range)
        print("file created!")

        return spectrum

    all_filename_to_create = []
    if batch_size:
        min_index = 0
        max_index = 0
        count = 0
    for index in range(flux_range_size):
        params.setHamiltonianLabels(index)
        filename_spectrum_flux = FM.filename_spectrum_eigenstates(params.MminLaughlin, params.MmaxLaughlin,
                                                                  params.edge_states, params.N, params.lz_val,
                                                                  params.hamiltonian_labels, params.h_parameters)
        # print(filename_spectrum_flux)
        if EC.does_file_really_exist(filename_spectrum_flux):
            # print("we did this calculation already")
            a = 1
        else:
            if send_jobs:
                if not batch_size:
                    sleep(2)
                    print("finding the spectrum for flux index=" + str(index))
                    args = [params_filename, index]
                    args = args + ['spectrum_eigenstates_manager']
                    job_name = 'spectrum_job_manager_' + "magnetic_flux_index=" + str(index) + "_FM_parameter=" + str(
                        params.FM_parameter)
                    job_name = EC.make_job_name_short_again(job_name)
                    queue = params.spectrum_job_manager_queue
                    mem, vmem = JS.get_mem_vmem_vals(queue, params.spectrum_job_manager_mem,
                                                     params.spectrum_job_manager_vmem)
                    JS.send_job(scriptNames.multiFile, pbs_filename=job_name, script_args=args,
                                queue=queue, mem=mem, vmem=vmem, ppn=params.ppn)
                else:
                    count = count + 1
                    max_index = index
                    if count == 1:
                        min_index = index
                        if index == (flux_range_size - 1):
                            # max_index = index
                            sleep(2)
                            print(
                                "sending a batch with FM=" + str(params.FM_parameter) + " and max_index=" + str(
                                    max_index))
                            args = [params_filename, min_index, max_index]
                            args = args + ['spectrum_eigenstates_batch']
                            job_name = 'flux_spectrum_batch_' + "min_flux_index=" + str(
                                min_index) + "_FM_parameter=" + str(
                                params.FM_parameter)
                            job_name = EC.make_job_name_short_again(job_name)
                            queue = params.spectrum_job_manager_queue
                            mem, vmem = JS.get_mem_vmem_vals(queue, params.spectrum_job_manager_mem,
                                                             params.spectrum_job_manager_vmem)
                            JS.send_job(scriptNames.multiFile, pbs_filename=job_name,
                                        script_args=args, queue=queue, mem=mem, vmem=vmem)
                    elif count == batch_size or index == (flux_range_size - 1):
                        # max_index = index
                        sleep(2)
                        print(
                            "sending a batch with FM=" + str(params.FM_parameter) + " and max_index=" + str(max_index))
                        args = [params_filename, min_index, max_index]
                        args = args + ['spectrum_eigenstates_batch']
                        job_name = 'flux_spectrum_batch_' + "min_flux_index=" + str(min_index) + "_FM_parameter=" + str(
                            params.FM_parameter)
                        job_name = EC.make_job_name_short_again(job_name)
                        queue = params.spectrum_job_manager_queue
                        mem, vmem = JS.get_mem_vmem_vals(queue, params.spectrum_job_manager_mem,
                                                         params.spectrum_job_manager_vmem)
                        JS.send_job(scriptNames.multiFile, pbs_filename=job_name, script_args=args,
                                    queue=queue, mem=mem, vmem=vmem)
                        count = 0

            all_filename_to_create.append(filename_spectrum_flux)
    if batch_size and count > 0:
        sleep(2)
        print("sending a batch with FM=" + str(params.FM_parameter) + " and max_index=" + str(max_index))
        args = [params_filename, min_index, max_index]
        args = args + ['spectrum_eigenstates_batch']
        job_name = 'flux_spectrum_batch_' + "min_flux_index=" + str(min_index) + "_FM_parameter=" + str(
            params.FM_parameter)
        job_name = EC.make_job_name_short_again(job_name)
        queue = params.spectrum_job_manager_queue
        mem, vmem = JS.get_mem_vmem_vals(queue, params.spectrum_job_manager_mem,
                                         params.spectrum_job_manager_vmem)
        JS.send_job(scriptNames.multiFile, pbs_filename=job_name,
                    script_args=args, queue=queue, mem=mem, vmem=vmem)

    while not EC.all_files_exist(all_filename_to_create):
        if not wait:
            print("still must create #" + str(len(all_filename_to_create)) + " files")
            return
        sleep(100)

    full_spectrum = {}

    print("writing spectrum into file " + filename_spectrum_vs_flux_range)
    for index in range(flux_range_size):
        # print(index)
        # print(params.potential_type)
        params.setHamiltonianLabels(index)
        # print(params.hamiltonian_labels)
        spec_states = FM.read_spectrum_eigenstates(params.MminLaughlin, params.MmaxLaughlin,
                                                   params.edge_states, params.N, params.lz_val,
                                                   params.hamiltonian_labels, params.h_parameters)
        spectrum = np.array([spec_states[i][0] for i in range(len(spec_states))])
        # print("***************")
        # print(spectrum)
        full_spectrum[params.flux_range[index]] = spectrum

    title = 'spectrum vs. magnetic flux of system with N=' + str(params.N) + ' MminL=' + str(
        params.MminLaughlin) + ' MmaxL=' + str(params.MmaxLaughlin) + '\nham_lbls: ' + ' '.join(
        params.hamiltonian_labels) + '\nLz laughlin at ' + str(params.lz_laughlin)
    xlabel = 'Magnetic flux'
    ylabel = 'Energy'

    graphData.write_spectrum_data_to_file(filename_spectrum_vs_flux_range, full_spectrum, title, xlabel, ylabel)

    return full_spectrum


def calc_luttinger_parameter(params_filename):
    params = ParametersAnnulus(params_filename)
    args = [params_filename, 'calc_luttinger_parameter']

    filename_luttinger = FM.filename_spectrum_luttinger_parm(params.MminLaughlin, params.MmaxLaughlin,
                                                             params.edge_states, params.N, 'all',
                                                             params.hamiltonian_labels, params.h_parameters)
    if EC.does_file_really_exist(filename_luttinger):
        print("we did this luttinger spectrum already")
        return

    job_name = 'lutt_p_N=' + str(params.N) + '_MminL=' + str(params.MminLaughlin) + '_MmaxL=' + str(
        params.MmaxLaughlin) + '_edges=' + str(params.edge_states)
    job_name = EC.make_job_name_short_again(job_name)
    queue = params.luttinger_parm_queue
    mem, vmem = JS.get_mem_vmem_vals(queue, params.luttinger_parm_mem, params.luttinger_parm_vmem)
    JS.send_job(scriptNames.multiFile, pbs_filename=job_name, script_args=args,
                queue=queue, mem=mem, vmem=vmem)


def calc_density_profile(params_filename):
    params = ParametersAnnulus(params_filename)
    args = [params_filename, 'calc_density_profile']
    job_name = 'density_profile_N=' + str(params.N) + '_MminL=' + str(params.MminLaughlin) + '_MmaxL=' + str(
        params.MmaxLaughlin) + '_edges=' + str(params.edge_states)
    job_name = EC.make_job_name_short_again(job_name)
    queue = params.managing_job_queue
    mem, vmem = JS.get_mem_vmem_vals(queue, params.managing_job_mem, params.managing_job_vmem)
    JS.send_job(scriptNames.multiFile, pbs_filename=job_name, script_args=args,
                queue=queue, mem=mem, vmem=vmem)
    return


def calc_luttinger_spectrum_big(params_filename, send_jobs=True, wait=False):
    params = ParametersAnnulus(params_filename)
    window_of_lz = 'all'

    filename_luttinger_spectrum = FM.filename_spectrum_luttinger_parm(params.MminLaughlin, params.MmaxLaughlin,
                                                                      params.edge_states,
                                                                      params.N, window_of_lz, params.hamiltonian_labels,
                                                                      params.h_parameters)
    if EC.does_file_really_exist(filename_luttinger_spectrum):
        print("full lut spectrum created already!")
        full_spectrum = graphData.plot_spectrum_graph_data_from_file(filename_luttinger_spectrum)
        return full_spectrum

    queue = params.spectrum_job_manager_queue
    mem, vmem = JS.get_mem_vmem_vals(queue, params.spectrum_job_manager_mem, params.spectrum_job_manager_vmem)
    Mmin = params.MminLaughlin - params.edge_states
    Mmax = params.MmaxLaughlin + params.edge_states

    first_arc_lz = np.array([params.lz_laughlin + i for i in range(params.N)])
    umbrella_lz = np.array(
        [params.lz_laughlin - params.N * params.edge_states + params.N * i for i in range(2 * params.edge_states + 1)])

    lz_vals_to_add_spectrum = np.concatenate((first_arc_lz, umbrella_lz))

    all_file_names = []
    for lz in lz_vals_to_add_spectrum:
        lz = int(lz)
        filename_lz_spectrum = FM.filename_spectrum_eigenstates(params.MminLaughlin, params.MmaxLaughlin,
                                                                params.edge_states, params.N, lz,
                                                                params.hamiltonian_labels, params.h_parameters)
        if EC.does_file_really_exist(filename_lz_spectrum):
            spectrum = FM.read_spectrum_eigenstates(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                    params.N, lz, params.hamiltonian_labels,
                                                    params.h_parameters)
            size_hilbert_space = GA.size_of_hilbert_space(Mmin, Mmax, params.N, lz)
            if len(spectrum) >= params.num_of_eigstates or size_hilbert_space <= params.num_of_eigstates:
                # print("we did this lz spectrum calculation already!")
                a = "do nothing"
            else:
                if send_jobs:
                    sleep(2)
                    print("finding the spectrum")
                    args = [params_filename, lz]
                    args = args + ['spectrum_eigenstates_manager']
                    # job_name = 'spectrum_job_manager_' + params.params_filename_no_path
                    job_name = 'lut_spectrum_N=' + str(params.N) + '_MminL=' + str(
                        params.MminLaughlin) + '_edges=' + str(params.edge_states) + '_lz=' + str(lz)
                    job_name = EC.make_job_name_short_again(job_name)
                    JS.send_job(scriptNames.multiFile, pbs_filename=job_name, script_args=args,
                                queue=queue, mem=mem, vmem=vmem)

                all_file_names.append(filename_lz_spectrum)

        else:
            if send_jobs:
                sleep(2)
                print("finding the spectrum")
                args = [params_filename, lz]
                args = args + ['spectrum_eigenstates_manager']
                # job_name = 'spectrum_job_manager_' + params.params_filename_no_path
                job_name = 'lut_spectrum_N=' + str(params.N) + '_MminL=' + str(
                    params.MminLaughlin) + '_edges=' + str(params.edge_states) + '_lz=' + str(lz)
                job_name = EC.make_job_name_short_again(job_name)
                JS.send_job(scriptNames.multiFile, pbs_filename=job_name, script_args=args,
                            queue=queue, mem=mem, vmem=vmem)

            all_file_names.append(filename_lz_spectrum)

    while not EC.all_files_exist(all_file_names):
        if not wait:
            print("not done creating spectrum! still must create " + str(len(all_file_names)) + " out of " + str(
                len(lz_vals_to_add_spectrum)))
            return
        sleep(100)

    lut_spectrum = {}
    for lz in lz_vals_to_add_spectrum:
        lz = int(lz)
        spec_states = FM.read_spectrum_eigenstates(params.MminLaughlin, params.MmaxLaughlin, params.edge_states,
                                                   params.N, lz, params.hamiltonian_labels, params.h_parameters)
        spectrum = np.array([spec_states[i][0] for i in range(len(spec_states))])
        print("***************")
        print(spectrum)
        lut_spectrum[lz] = spectrum

    title = 'Luttinger spectrum of system with N=' + str(params.N) + ' MminL=' + str(
        params.MminLaughlin) + ' MmaxL=' + str(params.MmaxLaughlin) + '\nham_lbls: ' + ' '.join(
        params.hamiltonian_labels) + '\nLz laughlin at ' + str(params.lz_laughlin)
    xlabel = 'Lz total'
    ylabel = 'Energy'
    graphData.write_spectrum_data_to_file(filename_luttinger_spectrum, lut_spectrum, title, xlabel, ylabel)

    return lut_spectrum

