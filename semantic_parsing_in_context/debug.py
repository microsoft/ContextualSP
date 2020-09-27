import json
import shutil
import sys

from allennlp.commands import main

if __name__ == '__main__':
    serialization_dir = "checkpoints/debug_model"
    config_file = "train_configs_bert/concat.none.mem.jsonnet"

    overrides = json.dumps({
        "dataset_reader.tables_file": "dataset_sparc/tables.json",
        "dataset_reader.database_path": "dataset_sparc/database",
        "train_data_path": "dataset_sparc/train.json",
        "validation_data_path": "dataset_sparc/dev.json",
        "model.dataset_path": "dataset_sparc",
        "model.serialization_dir": serialization_dir,
    })

    # Training will fail if the serialization directory already
    # has stuff in it. If you are running the same training loop
    # over and over again for debugging purposes, it will.
    # Hence we wipe it out in advance.
    # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
    shutil.rmtree(serialization_dir, ignore_errors=True)

    # in debug mode.
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", serialization_dir,
        "-f",
        "--include-package", "dataset_reader.sparc_reader",
        "--include-package", "models.sparc_parser",
        "-o", overrides
    ]

    main()
