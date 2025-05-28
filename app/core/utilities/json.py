import json
import os
import platform
from pathlib import Path

from six import string_types
from six.moves import collections_abc


# region conversion functions
def convert_str_to_dict(string):
    jstring = json.loads(string)
    return convert_to_dict(jstring)


def convert_dict_to_str(input_dict):
    return json.dumps(input_dict)


def convert_to_dict(data):
    if isinstance(data, string_types):
        return str(data)
    elif isinstance(data, collections_abc.Mapping):
        return dict(map(convert_to_dict, data.items()))
    elif isinstance(data, collections_abc.Iterable):
        return type(data)(map(convert_to_dict, data))
    else:
        return data


# endregion


# region load / dump json file
def dump_json(full_path, data):
    system = platform.system()
    if system == "Linux":
        full_path = full_path
        if not full_path.endswith(".json") and not full_path.endswith(".geojson"):
            full_path += ".json"
    elif system == "Windows":
        full_path = '\\\\?\\' + full_path
        if not full_path.endswith(".json") and not full_path.endswith(".geojson"):
            full_path += ".json"

    with open(full_path, 'w') as data_file:
        json.dump(data, data_file, indent=4, sort_keys=True)


def load_json(conf_path):
    with open(conf_path) as data_file:
        data = json.load(data_file)
    dict_data = convert_to_dict(data)
    return dict_data


# endregion

def pretty_print_dict(message, in_dict):
    pretty_json_data = json.dumps(convert_to_dict(in_dict), indent=4, sort_keys=True)
    return f'\n{message}:\n{pretty_json_data}'


def dump_json_exec_args(args, output_json_path='parsed_args.json'):
    args_dict = vars(args)
    os.makedirs(str(Path(output_json_path).parent), exist_ok=True)
    with open(output_json_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
    return args_dict


def str2bool(value):
    return False if value.lower() == 'false' else True
