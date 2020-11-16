from ATLASClusterInterface import JobSender as JS
from clusterScripts import scriptNames
import os
from time import sleep
import yaml

file = open('configurations.yml', 'r')
docs = yaml.full_load(file)
file.close()
directories = docs['directories']


def how_many_jobs(username=docs['username']):
    table_of_all_jobs = os.popen('qstat -u ' + username).read()
    list_jobs = table_of_all_jobs.split('\n')
    job_lines = list_jobs[5:-1]
    return len(job_lines)


def get_running_job_records(username=docs['username']):
    table_of_all_jobs = os.popen('qstat -u ' + username).read()
    list_jobs = table_of_all_jobs.split('\n')
    job_records = list_jobs[5:-1]
    job_records_broken = [l.split(" ") for l in job_records]
    for i in range(len(job_records_broken)):
        rec = job_records_broken[i]
        rec = [r for r in rec if r != '']
        job_records_broken[i] = rec
    return job_records_broken


def get_full_information_on_job(job_number):
    full_record = os.popen('qstat -f ' + job_number).read()
    full_record = full_record.split("\n")
    return full_record


def get_job_name(broken_job_record):
    job_number = broken_job_record[0][:8]
    full_record = get_full_information_on_job(job_number)
    if len(full_record) <= 1:
        print("job probably already finished running")
        return 0
    job_name_line = full_record[1]
    job_name = job_name_line[15:]
    return job_name


def get_matrix_pieces_running_jobs(matrix_name, args):
    running_job_records = get_running_job_records()
    running_job_names = [get_job_name(job) for job in running_job_records]
    running_job_names = [job for job in running_job_names if job != 0]
    running_job_names = [jn.split("-") for jn in running_job_names]
    matrix_details = [matrix_name] + args
    matrix_pieces_jobs = [job for job in running_job_names if job[:-2] == matrix_details]
    return matrix_pieces_jobs


def extract_failed_runs():
    stderr_dir = directories['std_error_dir']
    files_in_err_dir = os.listdir(stderr_dir)
    failed_jobs = [filename[:-10] for filename in files_in_err_dir if os.path.getsize(stderr_dir + filename) > 0]
    failed_jobs = [job.split("-") for job in failed_jobs]
    return failed_jobs


def failed_runs_of_matrix_pieces(matrix_name, args):
    args_list = [matrix_name] + [str(a) for a in args]
    all_failed_runs = extract_failed_runs()
    failed_runs = [job for job in all_failed_runs if job[:-2] == args_list]
    return failed_runs


def rerun_failed_matrix_piece(job):
    queue = 'N'
    mem = '4gb'
    vmem = '8gb'
    filename = "-".join(job)
    JS.send_job(scriptNames.piecesMBMatrix, queue, mem=mem, vmem=vmem, script_args=job, pbs_filename=filename)
    return 0


def is_job_still_running(job_name):
    sleep(1)
    running_jobs_records = get_running_job_records()
    running_jobs_names = [get_job_name(r) for r in running_jobs_records]
    if job_name in running_jobs_names:
        return True
    return False


def done_running_jobs(all_job_names):
    all_running_jobs = get_running_job_records()
    all_running_jobs_names = [get_job_name(rec) for rec in all_running_jobs]
    all_running_jobs_names = [jn for jn in all_running_jobs_names if jn != 0]
    for job_name in all_job_names:
        if job_name in all_running_jobs_names:
            return False
    return True


def job_failed(job_name):
    stderr_dir = directories['std_error_dir']
    files_in_err_dir = os.listdir(stderr_dir)
    failed_jobs = [filename[:-10] for filename in files_in_err_dir if os.path.getsize(stderr_dir + filename) > 0]
    if job_name in failed_jobs:
        print("job failed!")
        return True
    return False


def make_job_name_short_again(job_name):
    return job_name[0:50]


def does_file_really_exist(filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return True
    return False


def all_files_exist(file_names):
    for filename in file_names:
        if not does_file_really_exist(filename):
            return False
    return True
