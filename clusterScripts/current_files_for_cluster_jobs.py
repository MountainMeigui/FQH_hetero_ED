from DataManaging import fileManaging as FM

"""
send_off_spectrum_flux():
"""


filename_start = 'FM_parameter_scaling/spectrum_vs_flux_edge=2_N='
# Ns = ['7', '8', '9']
Ns = ['8']
filename_end = '_linear_m_space_flux=0.0001_FM_alt_spatial_flux_edge=0.002'
filenames = [filename_start + N + filename_end for N in Ns]

filename_start = 'testing_different_FM_terms/spectrum_vs_flux_edge=2_N=6_linear_m_space_flux=0.0001_FM_'
# FM_terms = ['spatial2', 'spatial6_flux']
# FM_terms = ['spatial3','spatialL3','spatialL2','spatial6L_flux','spatial4L_flux']
FM_terms = ['spatial_fixed1', 'spatial_fixed2', 'spatial_fixed_mixed']
filename_ends = ['0.001', '0.002', '0.003', '0.004', '0.005', '0.006', '0.007']
filenames1 = []
for filename_end in filename_ends:
    for FM_term in FM_terms:
        filenames1.append(filename_start + FM_term + filename_end)

filename_start = 'testing_different_FM_terms/spectrum_vs_flux_edge=3_N=6_linear_m_space_flux=0.0001_FM_'
# FM_terms = ['spatial2', 'spatial3', 'spatial6_flux', 'spatial4_flux']
# FM_terms = ['spatialL3','spatialL2','spatial6L_flux','spatial4L_flux']
# FM_terms = ['spatial_fixed1', 'spatial_fixed2']

filenames2 = []
for filename_end in filename_ends:
    for FM_term in FM_terms:
        filenames2.append(filename_start + FM_term + filename_end)

filenames = filenames1 + filenames2

# filename_start = 'FM_parameter_scaling/spectrum_vs_flux_edge=2_N='
# Ns = ['7', '8', '9']
# Ns = ['7']
# filename_end = '_linear_m_space_flux=0.0001_FM_alt_spatial_flux_edge=0.002'
# filenames = [filename_start + N + filename_end for N in Ns]

# filenames = ['spectrum_vs_flux_edge=2_N=6_linear_m_space_flux_conf_pot=0.0001_interactions']


"""
HISTORY
"""

"""
 full_spectrum_luttinger_test():
"""
# filename_p1 = 'luttinger_parm/luttinger_parm_calc_N='
filename_p1 = 'luttinger_parm/bigger_lut_parm_N='
filename_p2 = '_edges='
filename_p3 = '_MminL='
N_edges = [(6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (7, 2), (7, 3), (7, 4)]
filenames_start = [filename_p1 + str(N_edges[i][0]) + filename_p2 + str(N_edges[i][1]) + filename_p3 for i in
                   range(len(N_edges))]

"""
full_spectrum_for_thesis():
"""
filename = 'full_spectrum_graph_for_thesis_v4'
params_filename = FM.filename_parameters_annulus(filename)

"""
density_luttinger_calculations():
"""
filename_starts = ['luttinger_parm_calc_N=6_edges=2_MminL=', 'luttinger_parm_calc_N=6_edges=3_MminL=',
                   'luttinger_parm_calc_N=6_edges=4_MminL=', 'luttinger_parm_calc_N=6_edges=5_MminL=',
                   'luttinger_parm_calc_N=6_edges=6_MminL=', 'luttinger_parm_calc_N=7_edges=2_MminL=',
                   'luttinger_parm_calc_N=7_edges=3_MminL=',
                   'luttinger_parm/luttinger_parm_calc_N=8_edges=2_MminL=']
filename_end = [str(i) for i in range(5, 20)]

"""
send_test_luttinger_parm():
"""
# filename_start = 'luttinger_parm/testing_size_luttinger_parm_calc_MminL=10_edges=3_N='
# filename_start = 'luttinger_parm/testing_size_luttinger_parm_calc_MminL=10_edges=4_N='
# filename_start = 'luttinger_parm/testing_size_luttinger_parm_calc_MminL=10_edges=5_N='
filename_start = 'luttinger_parm/testing_size_luttinger_parm_calc_MminL=10_edges=6_N='
filenames = [filename_start + str(N) for N in range(7, 8)]

"""
send_luttinger_parm_calcs():
"""
# filename_start2 = 'luttinger_parm_calc_N=6_edges=5_MminL='
# filename_start3 = 'luttinger_parm_calc_N=6_edges=6_MminL='
# filename_start4 = 'luttinger_parm_calc_N=7_edges=4_MminL='
# filename_start = 'luttinger_parm/luttinger_parm_calc_N=8_edges=2_MminL='
# filename_start = 'luttinger_parm/luttinger_parm_calc_N=7_edges=4_MminL='
# filename_start = 'luttinger_parm/luttinger_parm_calc_N=7_edges=6_MminL='
# filename_start = 'luttinger_parm/luttinger_parm_calc_N=9_edges=2_MminL='
# filename_start = 'luttinger_parm/luttinger_parm_calc_N=7_edges=5_MminL='

# filename_end = [str(i) for i in range(5, 20)]
# filename_end = [str(i) for i in range(7, 27)]
# filename_end = [str(i) for i in range(7, 30)]
# filenames2 = [filename_start2 + end for end in filename_end]
# filenames3 = [filename_start3 + end for end in filename_end]
# filenames4 = [filename_start4 + end for end in filename_end]
# filenames = filenames2 + filenames3 + filenames4
# filenames = [filename_start + end for end in filename_end]
# filenames = [filename_start2 + '8', filename_start3 + '8', filename_start4 + '8']

# filename_p1 = 'luttinger_parm/luttinger_parm_calc_N='
filename_p1 = 'luttinger_parm/bigger_lut_parm_N='
filename_p2 = '_edges='
filename_p3 = '_MminL='
# N_edges = [(6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (8, 2), (8, 3), (9, 2)]
# N_edges = [(7, 5), (7, 6), (8, 2), (8, 3)]
# N_edges = [(9, 2)]
N_edges = [(7, 3), (7, 4)]
# N_edges = [(9, 2), (9, 3), (9, 4)]
N_edges = [(8, 4), (8, 5), (8, 6), (9, 2), (9, 3), (9, 4)]
# filename_end = [str(i) for i in range(7, 30)]
# N_edges = [(8, 5)]
# N_edges = [(8, 4)]
# N_edges = [(8, 5), (9, 2)]
N_edges = [(8, 6)]

# filename_end = [str(i) for i in range(11, 16)]
filename_end = [str(i) for i in range(19, 26)]

# N_edges = [(9, 2)]
# filename_end = [str(i) for i in range(10, 16)] + [str(i) for i in range(17, 24)]

# N_edges = [(9, 4)]
# filename_end = [str(i) for i in range(29, 30)]

# filename_end = ['10']

filenames_start = [filename_p1 + str(N_edges[i][0]) + filename_p2 + str(N_edges[i][1]) + filename_p3 for i in
                   range(len(N_edges))]

"""
send_off_spectrum_flux():
"""

filename_start = 'FM_parameter_scaling/spectrum_vs_flux_edge=2_N=6_linear_m_space_flux=0.0001_FM_alt_spatial_flux_edge='
# endings = ['0.002', '0.005', '0.001', '0.0001', '0.0005']
endings = ['0.0002']
endings = ['0.0001', '0.0002', '0.0003', '0.0004', '0.0005', '0.0006', '0.0007', '0.0008', '0.0009', '0.001',
           '0.0011', '0.0012', '0.0013', '0.0014', '0.0015', '0.0016', '0.0017', '0.0018', '0.0019', '0.002']
# endings = ['0.0021', '0.0022', '0.0023', '0.0024', '0.0025', '0.0026', '0.0027', '0.0028', '0.0029', '0.003',
#            '0.0031', '0.0032', '0.0033', '0.0034', '0.0035', '0.0036', '0.0037', '0.0038', '0.0039', '0.004',
#            '0.0041', '0.0042', '0.0043', '0.0044', '0.0045', '0.0046', '0.0047', '0.0048', '0.0049', '0.005']

# filename_start = 'FM_single_term/spectrum_vs_flux_edge=2_N=6_linear_m_space_flux=0.0001_FM_spatial1='
# endings = ['0.0001', '0.002', '0.005']
# endings = ['0.004', '0.01', '0.007']
filenames = [filename_start + end for end in endings]

filename_start = 'FM_parameter_scaling/spectrum_vs_flux_edge=2_N='
# Ns = ['7', '8', '9']
Ns = ['8']
filename_end = '_linear_m_space_flux=0.0001_FM_alt_spatial_flux_edge=0.002'
filenames = [filename_start + N + filename_end for N in Ns]

# filename_end1 = '=0.002'
# filename_end2 = '=0.003'
# filename_end2 = '=0.004'
filename_ends = ['0.001', '=0.002', '=0.003', '0.004', '0.005', '0.006', '0.007']

filename_start = 'testing_different_FM_terms/spectrum_vs_flux_edge=2_N=6_linear_m_space_flux=0.0001_FM_'
# FM_terms = ['spatial2', 'spatial6_flux']
# FM_terms = ['spatial3','spatialL3','spatialL2','spatial6L_flux','spatial4L_flux']
FM_terms = ['spatial_fixed1', 'spatial_fixed2', 'spatial_fixed_mixed']
filenames1 = []
for filename_end in filename_ends:
    for FM_term in FM_terms:
        filenames1.append(filename_start + FM_term + filename_end)

filename_start = 'testing_different_FM_terms/spectrum_vs_flux_edge=3_N=6_linear_m_space_flux=0.0001_FM_'
# FM_terms = ['spatial2', 'spatial3', 'spatial6_flux', 'spatial4_flux']
# FM_terms = ['spatialL3','spatialL2','spatial6L_flux','spatial4L_flux']
# FM_terms = ['spatial_fixed1', 'spatial_fixed2']

filenames2 = []
for filename_end in filename_ends:
    for FM_term in FM_terms:
        filenames2.append(filename_start + FM_term + filename_end)

filenames = filenames1 + filenames2

# filename_start = 'FM_parameter_scaling/spectrum_vs_flux_edge=2_N='
# Ns = ['7', '8', '9']
# Ns = ['7']
# filename_end = '_linear_m_space_flux=0.0001_FM_alt_spatial_flux_edge=0.002'
# filenames = [filename_start + N + filename_end for N in Ns]

# filenames = ['spectrum_vs_flux_edge=2_N=6_linear_m_space_flux_conf_pot=0.0001_interactions']


"""
calc_lz_spectrum(filename)
"""

files_to_calc = ['parms_for_FM_range_N=6_edge=2', 'parms_for_FM_range_N=6_edge=2_random',
                 'parms_for_FM_range_N=6_edge=1', 'parms_for_FM_range_N=6_edge=1_random',
                 'parms_for_FM_range_N=6_edge=2_random_smaller_FM', 'parms_for_FM_range_N=6_edge=2_smaller_FM']

# filename = 'lz_spectrum_for_IQH_flux_params_torus_like_conf_pot=0.001_interactions'


"""
calc_full_spectrum(filename)
"""

filename_start = 'luttinger_parm_calc_N=6_edges=2_MminL='
filename_end = [str(i) for i in range(5, 15)]
filenames = [filename_start + end for end in filename_end]
