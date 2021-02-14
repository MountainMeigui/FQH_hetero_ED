from time import sleep
from datetime import datetime
from os import environ
import yaml

file = open('../configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()
directories = docs['directories']
 

def limit_num_threads(num_threads=1):
    num_threads = str(num_threads)
    environ["OMP_NUM_THREADS"] = num_threads
    environ["OPENBLAS_NUM_THREADS"] = num_threads
    environ["MKL_NUM_THREADS"] = num_threads
    environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    environ["NUMEXPR_NUM_THREADS"] = num_threads
    return 0

