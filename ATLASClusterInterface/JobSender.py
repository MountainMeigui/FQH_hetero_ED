from subprocess import call
from time import sleep
from datetime import datetime
from os import environ
import yaml

file = open('configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()
directories = docs['directories']


def send_job(script, queue, **kargs):
    args = kargs.keys()
    # valid_args_names = ['run_dir', 'pbs_dir', 'pbs_filename', 'job_name', 'mem', 'vmem', 'err_dir', 'out_dir',
    #                     'script_args', 'ppn']
    if 'run_dir' in args:
        baseDirName = kargs['run_dir']
    else:
        baseDirName = directories['storage_dir_ATLAS']

    if 'pbs_dir' in args:
        pbs_dir = kargs['pbs_dir']
    else:
        pbs_dir = directories['PBS_files_dir']

    if 'pbs_filename' in args:
        filename = kargs['pbs_filename']
    else:
        filename = 'job' + str(datetime.now()).replace(" ", "_")
        # filename = 'job'
    pbs_filename = pbs_dir + filename + '.pbs'

    datastr = "#PBS -d " + baseDirName + " \n"
    datastr = datastr + "#PBS -V \n"
    if 'job_name' in args:
        job_name = kargs['job_name']
    else:
        job_name = filename
    datastr = datastr + "#PBS -N " + job_name + "\n"

    if 'ppn' in args:
        datastr = datastr + "#PBS -l nodes=1:ppn=" + str(kargs['ppn']) + " \n"
    else:
        datastr = datastr + "#PBS -l nodes=1:ppn=1 \n"

    if 'mem' in args:
        datastr = datastr + "#PBS -l mem=" + kargs['mem'] + " \n"
    if 'vmem' in args:
        datastr = datastr + "#PBS -l vmem=" + kargs['vmem'] + " \n"
    datastr = datastr + "#PBS -q " + queue + " \n"
    datastr = datastr + "#PBS -m n \n"
    if 'err_dir' in args:
        datastr = datastr + "#PBS -e " + kargs['err_dir'] + " \n"
    else:
        datastr = datastr + "#PBS -e " + directories['std_error_dir'] + " \n"

    if 'out_dir' in args:
        datastr = datastr + "#PBS -o " + kargs['out_dir'] + " \n"
    else:
        datastr = datastr + "#PBS -o " + directories['std_output_dir'] + " \n"

    datastr = datastr + directories['python_dir_ATLAS'] + "  " + directories['working_dir'] + str(script)

    if 'script_args' in args:
        script_args = " ".join([str(a) for a in kargs['script_args']])
        datastr = datastr + " " + script_args

    fo = open(pbs_filename, "w")
    fo.write(datastr)
    fo.close()
    correct_output = -1
    while correct_output != 0:
        try:
            correct_output = call(["qsub", pbs_filename])
            sleep(2)
        except:
            sleep(5)
        pass
    return 0


def limit_num_threads(num_threads=1):
    num_threads = str(num_threads)
    environ["OMP_NUM_THREADS"] = num_threads
    environ["OPENBLAS_NUM_THREADS"] = num_threads
    environ["MKL_NUM_THREADS"] = num_threads
    environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    environ["NUMEXPR_NUM_THREADS"] = num_threads
    return 0


def get_queue_max_mem_vals(queue):
    memory_max_values = {}

    if queue == 'S':
        memory_max_values['mem'] = '3048mb'
        memory_max_values['vmem'] = '6144mb'

    if queue == 'N':
        memory_max_values['mem'] = '4096mb'
        memory_max_values['vmem'] = '8192mb'

    if queue == 'P':
        memory_max_values['mem'] = '40gb'
        memory_max_values['vmem'] = '50gb'

    if queue == 'M':
        memory_max_values['mem'] = '500gb'
        memory_max_values['vmem'] = '520gb'

    return memory_max_values


def get_queue_default_mem_vals(queue):
    memory_def_values = {}

    if queue == 'S':
        memory_def_values['mem'] = '2048mb'
        memory_def_values['vmem'] = '4096mb'

    if queue == 'N':
        memory_def_values['mem'] = '2048mb'
        memory_def_values['vmem'] = '4096mb'

    if queue == 'P':
        memory_def_values['mem'] = '4096mb'
        memory_def_values['vmem'] = '4096mb'

    if queue == 'M':
        memory_def_values['mem'] = '4gb'
        memory_def_values['vmem'] = '4gb'

    return memory_def_values


def get_mem_vmem_vals(queue, mem, vmem):
    if mem == None:
        queue_mem_vals = get_queue_default_mem_vals(queue)
        mem = queue_mem_vals['mem']

    if vmem == None:
        queue_mem_vals = get_queue_default_mem_vals(queue)
        vmem = queue_mem_vals['vmem']

    if mem == 'max':
        queue_mem_vals = get_queue_max_mem_vals(queue)
        mem = queue_mem_vals['mem']

    if vmem == 'max':
        queue_mem_vals = get_queue_max_mem_vals(queue)
        vmem = queue_mem_vals['vmem']

    return mem, vmem
