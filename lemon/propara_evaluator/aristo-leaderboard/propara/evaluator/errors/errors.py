import sys


def corrupted_action_file(filename: str, details: str, line_num: int = None):
    if line_num is None:
        print(f"Corrupted or empty action file {filename} ({details})")
    else:
        print(f"Corrupted action file {filename} on line {line_num} ({details})")
    sys.exit(2)


def corrupted_sentences_file(filename: str, details: str):
    print(f"Corrupted or empty sentences file {filename} ({details})")
    sys.exit(2)
