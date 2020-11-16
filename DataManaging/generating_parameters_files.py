import yaml
from DataManaging import fileManaging as FM
from shutil import copyfile


def create_generic_parameters_file_for_annulus_geo(filename, MminL, MmaxL, edge_states, N, lz_val, potential_type,
                                                   interaction_parameter, confining_potential_name,
                                                   confining_potential_parameter, FM_term_name, FM_parameter):
    params_dict = {}
    params_dict['MminL'] = MminL
    params_dict['MmaxL'] = MmaxL
    params_dict['edge_states'] = edge_states
    params_dict['N'] = N
    params_dict['lz_val'] = lz_val
    params_dict['potential_type'] = potential_type
    params_dict['interaction_parameter'] = interaction_parameter
    params_dict['confining_potential_name'] = confining_potential_name
    params_dict['confining_potential_parameter'] = confining_potential_parameter
    params_dict['FM_term_name'] = FM_term_name
    params_dict['FM_parameter'] = FM_parameter

    # creating generic run parameters - values used in scripts
    background_params = {}
    short_filename = FM.get_short_parameters_filename(filename)
    background_params['params_full_filename'] = filename
    background_params['params_filename_no_path'] = short_filename

    """
    PARAMETERS FOR MatricesAndSpectra
    """

    if N <= 8:
        background_params['speeding_parameter'] = 3
    else:
        background_params['speeding_parameter'] = 10

    background_params['max_jobs_in_queue_S'] = 1800
    background_params['num_of_eigstates'] = 500

    background_params['write_unite_matrix_queue'] = 'N'
    background_params['write_unite_matrix_mem'] = None
    background_params['write_unite_matrix_vmem'] = None

    background_params['matrix_pieces_queue'] = 'S'
    background_params['matrix_pieces_mem'] = None
    background_params['matrix_pieces_vmem'] = None

    background_params['total_ang_momentum_label'] = 'None'
    """
    PARAMETERS FOR MatricesAndSpectraWapper
    """

    background_params['basis_list_queue'] = 'S'
    background_params['basis_list_mem'] = '1gb'
    background_params['basis_list_vmem'] = None

    background_params['two_particle_queue'] = 'S'
    background_params['two_particle_mem'] = '1gb'
    background_params['two_particle_vmem'] = None

    background_params['spectrum_eigenstates_queue'] = 'N'
    background_params['spectrum_eigenstates_mem'] = None
    background_params['spectrum_eigenstates_vmem'] = None

    background_params['spectrum_job_manager_queue'] = 'N'
    background_params['spectrum_job_manager_mem'] = '1gb'
    background_params['spectrum_job_manager_vmem'] = None

    background_params['lz_spectrum_manager_queue'] = 'N'
    # lz_spectrum_manager_mem = '1500mb'
    background_params['lz_spectrum_manager_mem'] = '2gb'
    background_params['lz_spectrum_manager_vmem'] = None

    background_params['lz_spectrum_queue'] = 'N'
    background_params['lz_spectrum_mem'] = '2gb'
    background_params['lz_spectrum_vmem'] = None

    background_params['managing_job_queue'] = 'N'
    background_params['managing_job_mem'] = '1gb'
    background_params['managing_job_vmem'] = None

    background_params['luttinger_parm_queue'] = 'S'
    background_params['luttinger_parm_mem'] = '1gb'
    background_params['luttinger_parm_vmem'] = None

    """
    PARAMETERS FOR create_parliminary_files_cluster_runs
    """

    background_params['is_hilbert_space_small'] = 1

    docs = {'primary_params': params_dict, 'background_params': background_params}
    file = open(filename, 'w')
    yaml.dump(docs, file)
    file.close()

    return docs


def change_params_file_for_annulus_geo(filename, to_be_changed, primary_or_back='p'):
    num_of_parameters = int(len(to_be_changed) / 2)
    file = open(filename, 'r')
    # docs = yaml.full_load(file)
    # docs = yaml.load(file, Loader=yaml.FullLoader)
    docs = yaml.load(file)
    file.close()
    if primary_or_back == 'p':
        dict_name = 'primary_params'
        # params_dict = docs['primary_params']
    else:
        dict_name = 'background_params'
        # params_dict = docs['background_params']
    for i in range(num_of_parameters):
        docs[dict_name][to_be_changed[2 * i]] = to_be_changed[2 * i + 1]

    file = open(filename, 'w')
    yaml.dump(docs, file)
    file.close()

    return docs


def copy_params_file(original_file_short, dest_name_short):
    original_filename = FM.filename_parameters_annulus(original_file_short)
    dest_name = FM.filename_parameters_annulus(dest_name_short)

    copyfile(original_filename, dest_name)

    change_params_file_for_annulus_geo(dest_name,
                                       ['params_full_filename', dest_name, 'params_filename_no_path', dest_name_short],
                                       'b')
    return dest_name


def update_parm_file(filename):
    luttinger_parm_addition = ['luttinger_parm_queue', 'S', 'luttinger_parm_mem', '1gb', 'luttinger_parm_vmem', None]
    change_params_file_for_annulus_geo(filename, luttinger_parm_addition,'b')
    return
