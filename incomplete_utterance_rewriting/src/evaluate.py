import argparse
import sys

from allennlp.commands import main

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_file", required=True, type=str,
                            help="Please specify a model file to evaluate")
    arg_parser.add_argument("--test_file", required=True, type=str,
                            help="Please specify a model file to evaluate")
    parsed_args = arg_parser.parse_args()

    model_file = parsed_args.model_file
    test_file = parsed_args.test_file
    result_file = model_file + ".json"

    sys.argv = [
        "allennlp",
        "evaluate",
        "--output-file", result_file,
        "--cuda-device", 0,
        "--include-package", "data_reader",
        "--include-package", "model",
        model_file,
        test_file
    ]

    main()
