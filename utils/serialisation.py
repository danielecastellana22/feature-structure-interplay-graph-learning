import json
import yaml
import torch as th
import pickle


def to_json_file(x, file_path):
    return json.dump(x, open(file_path, 'w'), indent='\t')


def from_json_file(file_path):
    return json.load(open(file_path, 'r'))


def to_yaml_file(x, file_path):
    return yaml.dump(x, open(file_path, 'w'), Dumper=yaml.SafeDumper, sort_keys=False)


def from_yaml_file(file_path):
    return yaml.load(open(file_path, 'r'), Loader=yaml.SafeLoader)


def to_pkl_file(x, file_path):
    return pickle.dump(x, open(file_path, 'wb'))


def from_pkl_file(file_path):
    return pickle.load(open(file_path, 'rb'))


def to_torch_file(x, file_path):
    return th.save(x, file_path)


def from_torch_file(file_path):
    return th.load(file_path)