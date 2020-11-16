import os
import sys

import yaml

file = open('configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()

if os.getcwd()[0] == '/':
    working_dir = docs['directories']['working_dir']
    sys.path.append(working_dir)


def delete_files_with_str_from_dir(string_in_filesname, dir_full_path):
    files_in_dir = os.listdir(dir_full_path)
    count = 0
    for filename in files_in_dir:
        if string_in_filesname in filename:
            count = count + 1
            os.remove(dir_full_path + filename)
    print("removed "+str(count)+" files from "+dir_full_path)
    return count
