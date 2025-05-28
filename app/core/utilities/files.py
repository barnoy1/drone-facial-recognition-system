import glob
import os
import shutil
from pathlib import Path

import pandas as pd
from tabulate import tabulate


def generate_from_list_csv_report(target_list, col_name='', report_full_path='report.csv'):
    df = pd.DataFrame(target_list, columns=[col_name])
    if len(df) == 0: return None
    generate_directory_if_not_exists(Path(report_full_path).parent)
    df.to_csv(report_full_path)
    status = (f'report path: {report_full_path}:\n '
              f'{tabulate(df, header="keys", tablefmt="psql", starlign="left", numalign="center")}')
    return status


def find_files_with_suffix(root_dir, suffix):
    search_pattern = os.path.join(root_dir, '**', f'*.{suffix}')
    found_files = glob.glob(search_pattern, recursive=True)
    return found_files


def delete_directory(path):
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)


def move_directory(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        shutil.move(s, d)


def normalize_string(str):
    str = str \
        .replace(' ', '_') \
        .replace('-', '') \
        .lower()
    return str


def copy_tree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copyfile(s, d)
