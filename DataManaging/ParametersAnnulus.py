import yaml
import numpy as np
from DataManaging import fileManaging as FM


class ParametersAnnulus:
    def __init__(self, filename):
        file = open(filename, 'r')
        # docs = yaml.full_load(file)
        # docs = yaml.load(file, Loader=yaml.FullLoader)
        docs = yaml.load(file)

        file.close()
        params_dict = docs['primary_params']

        self.MminLaughlin = params_dict['MminL']
        self.MmaxLaughlin = params_dict['MmaxL']
        self.edge_states = params_dict['edge_states']
        self.N = params_dict['N']
        self.lz_val = params_dict['lz_val']
        self.potential_type = params_dict['potential_type']
        self.interaction_parameter = params_dict['interaction_parameter']
        self.confining_potential_name = params_dict['confining_potential_name']
        self.confining_potential_parameter = params_dict['confining_potential_parameter']
        self.FM_term_name = params_dict['FM_term_name']
        self.FM_parameter = params_dict['FM_parameter']

        background_params = docs['background_params']

        self.speeding_parameter = background_params['speeding_parameter']
        self.max_jobs_in_queue_S = background_params['max_jobs_in_queue_S']
        self.num_of_eigstates = background_params['num_of_eigstates']

        self.write_unite_matrix_queue = background_params['write_unite_matrix_queue']
        self.write_unite_matrix_mem = background_params['write_unite_matrix_mem']
        self.write_unite_matrix_vmem = background_params['write_unite_matrix_vmem']

        self.matrix_pieces_queue = background_params['matrix_pieces_queue']
        self.matrix_pieces_mem = background_params['matrix_pieces_mem']
        self.matrix_pieces_vmem = background_params['matrix_pieces_vmem']

        self.total_ang_momentum_label = background_params['total_ang_momentum_label']

        self.basis_list_queue = background_params['basis_list_queue']
        self.basis_list_mem = background_params['basis_list_mem']
        self.basis_list_vmem = background_params['basis_list_vmem']

        self.two_particle_queue = background_params['two_particle_queue']
        self.two_particle_mem = background_params['two_particle_mem']
        self.two_particle_vmem = background_params['two_particle_vmem']

        self.spectrum_eigenstates_queue = background_params['spectrum_eigenstates_queue']
        self.spectrum_eigenstates_mem = background_params['spectrum_eigenstates_mem']
        self.spectrum_eigenstates_vmem = background_params['spectrum_eigenstates_vmem']

        self.spectrum_job_manager_queue = background_params['spectrum_job_manager_queue']
        self.spectrum_job_manager_mem = background_params['spectrum_job_manager_mem']
        self.spectrum_job_manager_vmem = background_params['spectrum_job_manager_vmem']

        self.lz_spectrum_manager_queue = background_params['lz_spectrum_manager_queue']
        self.lz_spectrum_manager_mem = background_params['lz_spectrum_manager_mem']
        self.lz_spectrum_manager_vmem = background_params['lz_spectrum_manager_vmem']

        self.lz_spectrum_queue = background_params['lz_spectrum_queue']
        self.lz_spectrum_mem = background_params['lz_spectrum_mem']
        self.lz_spectrum_vmem = background_params['lz_spectrum_vmem']

        self.managing_job_queue = background_params['managing_job_queue']
        self.managing_job_mem = background_params['managing_job_mem']
        self.managing_job_vmem = background_params['managing_job_vmem']

        self.luttinger_parm_queue = background_params['luttinger_parm_queue']
        self.luttinger_parm_mem = background_params['luttinger_parm_mem']
        self.luttinger_parm_vmem = background_params['luttinger_parm_vmem']

        self.is_hilbert_space_small = background_params['is_hilbert_space_small']

        self.params_full_filename = filename
        self.params_filename_no_path = FM.get_short_parameters_filename(filename)

        # self.params_full_filename = background_params['params_full_filename']
        # self.params_filename_no_path = background_params['params_filename_no_path']
        # self.params_full_filename = FM.filename_parameters_annulus(self.params_filename_no_path)

        # more useful combinations
        self.hamiltonian_labels = [self.potential_type, self.confining_potential_name, 'None', self.FM_term_name]
        self.h_parameters = [self.interaction_parameter, self.confining_potential_parameter, 0.0, self.FM_parameter]
        self.m = 3
        self.lz_laughlin = self.m * self.N * (self.N - 1) / 2 + self.MminLaughlin * self.N
        self.is_magnetic_flux = 0
        self.ppn = 1

        # optional params
        if 'FM_parameter_range' in params_dict.keys():
            self.FM_parameter_range_vals = params_dict['FM_parameter_range']
            self.FM_parameter_range = np.arange(*params_dict['FM_parameter_range'])
        if 'magnetic_flux' in params_dict.keys():
            self.is_magnetic_flux = 1
            self.magnetic_flux = params_dict['magnetic_flux']
            self.hamiltonian_labels[0] = self.hamiltonian_labels[0] + '_' + str(self.magnetic_flux)
            self.hamiltonian_labels[1] = self.hamiltonian_labels[1] + '_' + str(self.magnetic_flux)
        if 'flux_range' in params_dict.keys():
            self.is_magnetic_flux = 1
            self.magnetic_flux = params_dict['flux_range'][0]
            self.flux_range_vals = params_dict['flux_range']
            self.flux_range = np.arange(*params_dict['flux_range'])
        if 'num_measurement_points' in background_params.keys():
            self.num_measurement_points = background_params['num_measurement_points']

        if 'eigenstates_cutoff' in background_params.keys():
            self.eigenstates_cutoff = background_params['eigenstates_cutoff']
        else:
            self.eigenstates_cutoff = None
        if 'ppn' in background_params.keys():
            self.ppn = background_params['ppn']

    def setHamiltonianLabels(self, magnetic_flux_index='no_flux'):
        if magnetic_flux_index == 'no_flux':
            self.hamiltonian_labels[0] = self.potential_type
            self.hamiltonian_labels[1] = self.confining_potential_name
            return

        # if magnetic_flux_index != 0:
        #     self.magnetic_flux = self.flux_range[magnetic_flux_index]
        self.magnetic_flux = self.flux_range[magnetic_flux_index]
        if self.hamiltonian_labels[0] != 'None':
            self.hamiltonian_labels[0] = self.potential_type + '_' + str(self.magnetic_flux)
        if self.hamiltonian_labels[1] != 'None':
            self.hamiltonian_labels[1] = self.confining_potential_name + '_' + str(self.magnetic_flux)
        if self.hamiltonian_labels[3] != 'None':
            self.hamiltonian_labels[3] = self.FM_term_name + '_' + str(self.magnetic_flux)
        return
