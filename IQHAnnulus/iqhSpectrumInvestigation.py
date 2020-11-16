import numpy as np
from AnnulusFQH import BasisAndMBNonInteracting as GA, singleParticleOperatorsOnAnnulus as SPA
from IQHAnnulus import nonInteractingMBHamiltonians as NIMB
import matplotlib.pyplot as plt


def spectrum_to_dictionary_by_lz(mb_spectrum, mb_lz_vals):
    full_spectrum_dict = {}
    min_lz = min(mb_lz_vals)
    max_lz = max(mb_lz_vals)
    for i in range(min_lz, max_lz + 1):
        full_spectrum_dict[i] = []

    for i in range(len(mb_spectrum)):
        full_spectrum_dict[mb_lz_vals[i]].append(mb_spectrum[i])

    for i in range(min_lz, max_lz + 1):
        full_spectrum_dict[i] = np.array(sorted(full_spectrum_dict[i]))

    return full_spectrum_dict


def calc_luttinger_parameter_from_full_spectrum(full_spectrum, lz_center_val, N, edge_states):
    first_arc_lz = np.array([lz_center_val + i for i in range(N)])
    first_arc_energy = np.array([full_spectrum[lz][0] for lz in first_arc_lz])
    plt.figure()
    plt.plot(first_arc_lz, first_arc_energy, '_')
    print(first_arc_lz)
    print(first_arc_energy)
    p_first_arc, V_first_arc = np.polyfit(first_arc_lz, first_arc_energy, 2, cov=True)
    omega = p_first_arc[0] * 2 * first_arc_lz[0] + p_first_arc[1]
    # half_N = int(N / 2 + 1)
    # p_first_arc, V_first_arc = np.polyfit(first_arc_lz[:half_N], first_arc_energy[:half_N], 1, cov=True)
    # omega = p_first_arc[0]
    print(omega)
    polyfitted = np.array([np.polyval(p_first_arc, lz) for lz in first_arc_lz])
    plt.plot(first_arc_lz, polyfitted)
    #
    plt.figure()
    umbrella_lz = np.array(
        [lz_center_val - N * edge_states + N * i for i in range(2 * edge_states + 1)])
    umbrella_energy = np.array([full_spectrum[lz][0] for lz in umbrella_lz])
    umbrella_J = np.array([2 * excitation for excitation in range(-edge_states, edge_states + 1)])
    plt.plot(umbrella_lz, umbrella_energy, '_')
    print(umbrella_J)
    print(umbrella_lz)
    print(umbrella_energy)
    p_umbrella, V_umbrella = np.polyfit(umbrella_J, umbrella_energy, 2, cov=True)
    polyfitted_u = np.array([np.polyval(p_umbrella, J) for J in umbrella_J])
    plt.plot(umbrella_lz, polyfitted_u)
    omega_J_over_four = p_umbrella[0]
    print("omega_J/4 = " + str(omega_J_over_four))

    # g = 4 * v_J_over_four_R / v_s_over_R
    omega2 = full_spectrum[lz_center_val + 1][0] - full_spectrum[lz_center_val][0]
    g2 = omega_J_over_four * 4 / omega2

    omega3 = [full_spectrum[lz_center_val + e * N + 1][0] - full_spectrum[lz_center_val + e * N][0] for e in
              range(-edge_states + 1, edge_states)]
    print(omega3)
    omega3 = sum(omega3) / len(omega3)
    g3 = omega_J_over_four * 4 / omega3

    g = omega_J_over_four * 4 / omega
    print("***********")
    print(g)
    print(g2)
    print(g3)
    plt.show()
    return g


N = 13
MminI = 10
MmaxI = MminI + N - 1
lz_integer = sum([k for k in range(MminI, MmaxI + 1)])
print(lz_integer)
edge_states = 4
Mmin = MminI - edge_states
Mmax = MmaxI + edge_states

conf_pot_single = SPA.create_single_particle_operator(MminI, MmaxI, edge_states, 'linear_m_space_flux',
                                                      'confining_potential')
# conf_pot_single = SPA.create_single_particle_operator(MminI, MmaxI, edge_states, 'space_parabolic_middle_flux',
#                                                       'confining_potential')

diag = [conf_pot_single[k, k] for k in range(conf_pot_single.shape[0])]
plt.figure()
plt.plot(diag, '.')
plt.figure()
lz_single = SPA.total_angular_momentum_single_particle(Mmin, Mmax)

eigenvals, eigenvecs = NIMB.get_single_particle_spectrum_states(conf_pot_single)

many_body_spectrum = NIMB.get_many_body_spectrum_from_single_particle_spectrum(eigenvals, N)
# print(many_body_spectrum)
many_body_lz = NIMB.get_observable_MB(eigenvals, eigenvecs, lz_single, N)
many_body_lz = np.array([int(val) for val in many_body_lz])
# print(many_body_lz)
full_spectrum_dict = spectrum_to_dictionary_by_lz(many_body_spectrum, many_body_lz)
# print(full_spectrum_dict)
plt.plot(many_body_lz, many_body_spectrum, '_')
# plt.show()
calc_luttinger_parameter_from_full_spectrum(full_spectrum_dict, lz_integer, N, edge_states)
