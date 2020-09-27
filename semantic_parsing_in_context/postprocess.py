# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import argparse


def convert_dataset(valid_file, valid_out_file):
    """
    The package `allennlp` requires the validation file as the format of each line containing a json object.
    :param valid_file: valid file input, the original dataset validation file.
    :param valid_out_file: valid file output, the adapted file for allennlp package.
    """
    write_file = open(valid_out_file, "w", encoding="utf8")
    with open(valid_file, "r", encoding="utf8") as f:
        content = json.load(f)
        for instance in content:
            write_file.write(json.dumps(instance) + "\n")
    write_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_file', type=str)
    parser.add_argument('--valid_out_file', type=str)
    args = parser.parse_args()
    convert_dataset(args.valid_file, args.valid_out_file)
