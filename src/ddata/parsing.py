import os
import pandas as pd
import pathlib
import tqdm

data_path = pathlib.Path('osfstorage-archive')




columns = ['trial_num', 'drift1', 'drift2', 'drift3', 'drift4', 'stage1_resp', 'stage1_select', 'stage1_rt', 'transition', 'stage2_resp', 'stage2_select', 'stage2_state', 'stage2_rt', 'reward', '_']

def parse_2step_lines(lines):
    lines = [line.rstrip().split(',') for line in lines]
    age = int(lines[0][1])
    gender = lines[1][1]
    subject_id = lines[2][0]
    start_line = -1
    for i in range(2, len(lines)):
        if len(lines[i]) == len(columns): # and lines[i][0] == "1":
            start_line = i
            break
    else:
        raise LookupError

    data = pd.DataFrame(lines[start_line:], columns=columns)
    return data

def parse_2step_file(file_path):
    with open(file_path) as file:
        try:
            return parse_2step_lines(file.read().split('\n'))
        except LookupError:
            raise Exception(f'Couldn\'t find matching start_line for file {file_path}')

# The first column for all these csv files is just 1,2,3,... (Unnamed index)

# self report study
srs1 = pd.read_csv(data_path / 'Experiment 1' / 'self_report_study1.csv')
srs2 = pd.read_csv(data_path / 'Experiment 2' / 'self_report_study2.csv')
srs1.drop(srs1.columns[0], axis=1, inplace=True)
srs2.drop(srs2.columns[0], axis=1, inplace=True)

ii2 = pd.read_csv(data_path / 'Experiment 2' / 'individual_items_study2.csv')
ii2.drop(ii2.columns[0], axis=1, inplace=True)
# make the subject column the first column
cols = ii2.columns.tolist()
subj_i = cols.index('subject')
ii2 = ii2[cols[subj_i:subj_i+1] + cols[:subj_i] + cols[subj_i+1:]]

def extract_2step(folder_path):
    files = os.listdir(folder_path)

    subj_rl_data = []
    for subj_rl_file in tqdm.tqdm(files):
        subj_rl_data.append([subj_rl_file, parse_2step_file(folder_path / subj_rl_file)])

    return subj_rl_data

# exp1_rl_data = extract_2step(data_path / 'Experiment 1' / 'twostep_data_study1')
# exp2_rl_data = extract_2step(data_path / 'Experiment 2' / 'twostep_data_study2')
#
