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

from AnnulusFQH import MatricesAndSpectra as AMAS
from DataManaging import fileManaging as FM
from ATLASClusterInterface import JobSenderFQH as JS

JS.limit_num_threads()

matrix_name = sys.argv[1]
MminL = int(sys.argv[2])
MmaxL = int(sys.argv[3])
edge_states = int(sys.argv[4])
N = int(sys.argv[5])
lz_val = sys.argv[6]
matrix_label = sys.argv[7]

if lz_val != 'not_fixed':
    lz_val = int(float(lz_val))

args = [MminL, MmaxL, edge_states, N, lz_val, matrix_label]
filename_complete_matrix = FM.filename_complete_matrix(matrix_name, args)
if FM.does_file_really_exist(filename_complete_matrix):
    print("matrix already written")
else:
    AMAS.unite_matrix_pieces(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name)
    AMAS.delete_excess_pieces(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name)
