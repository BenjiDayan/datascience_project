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
    for i in range(2, len(lines)):
        if len(lines[i]) > 2:
            if lines[i][-2] == 'twostep_instruct_9':
                start_line = i + 1
                break

    data = pd.DataFrame(lines[start_line:], columns=columns)
    return data

# self report study
srs1 = pd.read_csv(data_path / 'Experiment 1' / 'self_report_study1.csv')
srs2 = pd.read_csv(data_path / 'Experiment 2' / 'self_report_study2.csv')


def extract_2step(folder_path):
    files = os.listdir(folder_path)

    subj_rl_data = []
    for subj_rl_file in tqdm.tqdm(files):
        with open(data_path / subj_rl_file) as file:
            subj_rl_data.append(parse_2step_lines(file.read().split('\n')))

    return subj_rl_data

exp1_rl_data = extract_2step(data_path / 'Experiment 1' / 'twostep_data_study1')
exp2_rl_data = extract_2step(data_path / 'Experiment 2' / 'twostep_data_study2')

