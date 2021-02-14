import pickle
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from DataManaging import fileManaging as FM
import os

if os.getcwd()[0] != "/":
    matplotlib.use('Qt5Agg')


def write_spectrum_data_to_file(filename, spectrum, title, xlabel, ylabel):
    dic = {'spectrum': spectrum, 'title': title, 'xlabel': xlabel, 'ylabel': ylabel}
    file1 = open(filename, 'wb')
    pickle.dump(dic, file1, protocol=4)
    file1.close()
    if title:
        print("wrote " + title + ' into ' + filename)
    else:
        print("wrote graph into " + filename)
    return 0


def add_cutoff_energy_to_spectrum(full_spectrum, forced=0):
    natural_cutoff_energy_ind = min(full_spectrum, key=lambda x: full_spectrum[x][-1])
    natural_cutoff_energy = full_spectrum[natural_cutoff_energy_ind][-1]
    if forced:
        cutoff_energy = forced
    else:
        cutoff_energy = natural_cutoff_energy
    cut_spectrum = {}
    for lz_val in full_spectrum.keys():
        if full_spectrum[lz_val][0] <= cutoff_energy:
            lz_spec = np.array(full_spectrum[lz_val])
            cut_spectrum[lz_val] = lz_spec[lz_spec <= cutoff_energy]
    return cut_spectrum, cutoff_energy


def count_gs_degeneracies(full_spectrum, epsilon=10 ** -13):
    min_energy_ind = min(full_spectrum, key=lambda x: full_spectrum[x][0])
    min_energy = full_spectrum[min_energy_ind][0]
    print("minimal energy is " + str(min_energy))
    print("minimal energy at lz= " + str(min_energy_ind))
    count = 0
    for lz_val in full_spectrum.keys():
        for energy_val in full_spectrum[lz_val]:
            if abs(energy_val - min_energy) <= epsilon:
                count = count + 1
    return count


def count_all_states(spectrum):
    count = 0
    for lz_val in spectrum.keys():
        count = count + len(spectrum[lz_val])
    return count


def edge_states_count(edge_spectrum):
    states_count = {}
    for lz_val in edge_spectrum.keys():
        states_count[lz_val] = len(edge_spectrum[lz_val])
    print("range of edge states from " + str(min(edge_spectrum.keys())) + " to " + str(max(edge_spectrum.keys())))
    return states_count


def plot_spectrum_graph_data_from_file(filename, cutoff=None, force_cutoff=None):
    fontsize = 25
    file1 = open(filename, 'rb')
    data = pickle.load(file1)
    file1.close()
    color = 'xkcd:plum'

    spectrum = data['spectrum']
    if cutoff:
        if force_cutoff:
            spectrum, cutoff_energy = add_cutoff_energy_to_spectrum(spectrum, cutoff)
        else:
            spectrum, cutoff_energy = add_cutoff_energy_to_spectrum(spectrum)

    for val in spectrum.keys():
        spec_val = np.array(spectrum[val])
        plt.plot([val] * len(spec_val), spec_val, '_', markersize=7, color=color)

    if data['xlabel'] != None:
        plt.xlabel(data['xlabel'], fontsize=fontsize)
    if data['ylabel'] != None:
        plt.ylabel(data['ylabel'], fontsize=fontsize)
    if data['title']:
        if cutoff:
            plt.title(data['title'] + '\ncutoff energy is: ' + str(round(cutoff_energy, 2)))
        else:
            plt.title(data['title'])
    # plt.show()
    return spectrum


def write_graph_data_to_file(filename, x, y, title=None, legend=None, xlabel=None, ylabel=None):
    x = np.array(x)
    y = np.array(y)
    dic = {'x': x, 'y': y, 'title': title, 'legend': legend, 'xlabel': xlabel, 'ylabel': ylabel}
    file1 = open(filename, 'wb')
    pickle.dump(dic, file1, protocol=4)
    file1.close()
    if title:
        print("wrote " + title + ' into ' + filename)
    else:
        print("wrote graph into " + filename)

    return 0


def read_graph_data_from_file(filename, cutoff=None):
    file1 = open(filename, 'rb')
    data = pickle.load(file1)
    file1.close()
    x = data['x']
    # y = abs(data['y'])
    y = data['y']

    if cutoff:
        x = x[y <= cutoff]
        y = y[y <= cutoff]

    return x, y


def plot_graph_from_file(filename, cutoff=None):
    file1 = open(filename, 'rb')
    data = pickle.load(file1)
    file1.close()
    x = data['x']
    # y = abs(data['y'])
    y = data['y']

    if cutoff:
        x = x[y <= cutoff]
        y = y[y <= cutoff]

    if data['legend'] != None:
        plt.plot(x, y, '_', label=data['legend'])
    else:
        plt.plot(x, y, '_')
    if data['xlabel'] != None:
        plt.xlabel(data['xlabel'])
    if data['ylabel'] != None:
        plt.ylabel(data['ylabel'])
    if data['title'] != None:
        plt.title(data['title'])
    if data['legend'] != None:
        plt.legend()
    # plt.show()
    return x, y


def linear_fit_data(x, y):
    p, V = np.polyfit(x, y, 1, cov=True)
    slope = p[0]
    slope_err = np.sqrt(V[0][0])
    return slope, slope_err


def sort_low_lying_from_full_spectrum(full_spectrum):
    low_lying_spectrum = []
    max_energy_of_sense_lz_val = min(full_spectrum.keys(), key=(lambda x: full_spectrum[x][-1]))
    max_energy_of_sense = full_spectrum[max_energy_of_sense_lz_val][-1]

    lz_vals = full_spectrum.keys()
    for lz in lz_vals:
        spec_lz = np.array(full_spectrum[lz])
        relevant_energies = spec_lz[spec_lz <= max_energy_of_sense]
        energies_lz_val = [(lz, en) for en in relevant_energies]
        low_lying_spectrum = low_lying_spectrum + energies_lz_val

    low_lying_spectrum = sorted(low_lying_spectrum, key=(lambda x: x[1]))
    return low_lying_spectrum


def round_value_to_fit_lattice(value, lattice):
    lattice_spacing = lattice[1] - lattice[0]
    lattice_ind = int((value - lattice[0]) / lattice_spacing)
    return lattice_ind


def plot_energy_vs_lz_toggle(MminL, MmaxL, edge_states, N, ham_labels, interaction_strength, confining_strength,
                             FM_range, cutoff=None, subspace_size=0):
    filenameFunc = lambda x: FM.filename_spectrum_lz_total_vals(MminL, MmaxL, edge_states, N, ham_labels,
                                                                [interaction_strength, confining_strength, 0.0, x])
    if subspace_size:
        filenameFunc = lambda x: FM.filename_lz_total_spectrum_low_lying_subspace(MminL, MmaxL, edge_states, N,
                                                                                  ham_labels, [interaction_strength,
                                                                                               confining_strength, 0.0,
                                                                                               x], subspace_size)
    plot_data_vs_parameter_toggle(FM_range, filenameFunc, cutoff)


def plot_spectrum_for_FM_range(MminL, MmaxL, edge_states, N, ham_labels, interaction_strength, confining_strength,
                               FM_range, cutoff=None, ax=None, subspace_size=0):
    filenameFunc = lambda x: FM.filename_spectrum_lz_total_vals(MminL, MmaxL, edge_states, N, ham_labels,
                                                                [interaction_strength, confining_strength, 0.0, x])
    if subspace_size:
        filenameFunc = lambda x: FM.filename_lz_total_spectrum_low_lying_subspace(MminL, MmaxL, edge_states, N,
                                                                                  ham_labels, [interaction_strength,
                                                                                               confining_strength, 0.0,
                                                                                               x], subspace_size)

    title = 'Energy spectrum for MminL=' + str(MminL) + ' MmaxL=' + str(MmaxL) + ' edges=' + str(
        edge_states) + ' N= ' + str(N) + '\ninteraction type=' + ham_labels[0] + ' ' + str(
        interaction_strength) + '\nconfining potential=' + ham_labels[1] + ' ' + str(
        confining_strength) + '\nFM term=' + \
            ham_labels[3]
    xlabel = 'FM coupling strength'
    ylabel = 'Energy'

    plot_data_for_parameter_range(FM_range, filenameFunc, title, xlabel, ylabel, cutoff, ax)


"""
General functions for viewing data with a changing parameter
"""


def update_data_plot(g_plot, fig, parameter, lattice_parms, filenameFromParameter, cutoff):
    ind_in_lattice = round_value_to_fit_lattice(parameter, lattice_parms)
    p_discrete = lattice_parms[ind_in_lattice]
    filename_graph_data = filenameFromParameter(p_discrete)
    x, y = read_graph_data_from_file(filename_graph_data, cutoff)
    g_plot.set_xdata(x)
    g_plot.set_ydata(y)
    fig.canvas.draw_idle()


def plot_data_vs_parameter_toggle(lattice_parms, filenameFromParameter, cutoff=None):
    p_range_min = lattice_parms[0]
    p_range_max = lattice_parms[-1]
    p_range_init = lattice_parms[0]

    filename_graph_data = filenameFromParameter(p_range_init)
    x, y = read_graph_data_from_file(filename_graph_data, cutoff)

    fig = plt.figure(figsize=(8, 3))
    graph_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

    plt.axes(graph_ax)
    data_plot, = plt.plot(x, y, '_')

    # y_space = (max(y) - min(y)) / 10
    y_space = (max(y) - min(y)) * 4
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - y_space, max(y) + y_space)

    FM_slider = Slider(slider_ax, 'FM coupling strength', p_range_min, p_range_max, valinit=p_range_init)
    FM_slider.on_changed(
        lambda x: update_data_plot(data_plot, fig, x, lattice_parms, filenameFromParameter, cutoff))
    plt.show()


def plot_data_for_parameter_range(lattice_parms, filenameFromParameter, title='', xlabel='x', ylabel='y', cutoff=None,
                                  ax=None):
    grph = ax
    if not ax:
        grph = plt
        grph.figure()

    for parm in lattice_parms:
        filename_data_for_p = filenameFromParameter(parm)
        x, y = read_graph_data_from_file(filename_data_for_p, cutoff)
        grph.plot([parm] * len(y), y, '_')

    if ax:
        grph.set(xlabel=xlabel, ylabel=ylabel)
        grph.set_title(title)
    else:
        grph.xlabel(xlabel)
        grph.ylabel(ylabel)
        grph.title(title)
    if not ax:
        plt.show()
    return 0


def print_1D_map_of_func(z, func, funcName, label):
    density_profile = np.array([func(x) for x in z])
    r = np.array([abs(x) for x in z])

    if label:
        plt.plot(r, density_profile, '.', label=label)
    else:
        plt.plot(r, density_profile, '.')
    if funcName:
        plt.title(funcName)
    plt.xlabel('r')
    return density_profile
