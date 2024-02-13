from conf.data_config import DataConfig

import json


def find_mismatched_keys(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        data1 = json.load(file1)
        data2 = json.load(file2)

    mismatched_keys = []

    for key in data1:
        if key in data2 and data1[key]['is_correct'] != data2[key]['is_correct']:
            mismatched_keys.append(key)

    return mismatched_keys


file1_path = "data/second_iteration/two_step_gpt_results.json"
file2_path = DataConfig().two_step_gpt_results_file

mismatched_keys = find_mismatched_keys(file1_path, file2_path)

print("Keys with 'is_correct' mismatch:")
for key in mismatched_keys:
    print(key)
