import sys
import os
import yaml

file = open('../configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()

if os.getcwd()[0] == '/':
    working_dir = docs['directories']['working_dir']
    sys.path.append(working_dir)

from AnnulusFQH import MatricesAndSpectra as AMAS

matrix_name = sys.argv[1]
MminL = int(sys.argv[2])
MmaxL = int(sys.argv[3])
edge_states = int(sys.argv[4])
N = int(sys.argv[5])
lz_val = sys.argv[6]
matrix_label = sys.argv[7]

if lz_val != 'not_fixed':
    lz_val = int(float(lz_val))

AMAS.extract_complete_matrix(MminL, MmaxL, edge_states, N, lz_val, matrix_label, matrix_name, 0)
