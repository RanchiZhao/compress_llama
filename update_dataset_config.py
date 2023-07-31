import json
import os
import bmtrain as bmt
import sys

def update_dataset_config(dataset_path):
    if bmt.rank() != 0:
        return
    platform_config_path = os.getenv("PLATFORM_CONFIG_PATH")
    with open(dataset_path, "r") as f:
        cfg = json.load(f)
    with open(platform_config_path, "r") as f:
        platform_cfg = json.load(f)
    path_dict = platform_cfg["dataset_map"]

    path_dict = {
    "lm_eng": "/path/to/lm_eng/dataset",
    "pile_v3": "/path/to/pile_v3/dataset",
    # ... other datasets ...
    }

    for dataset in cfg:
        if 'data_pack' in dataset:
            dataset["path"] = os.path.join(path_dict[dataset["data_pack"]], dataset["path"])
            transforms = dataset.get("transforms", None)
            if transforms is not None:
                dataset["transforms"] = os.path.join(path_dict[dataset["data_pack"]], dataset["transforms"])
        else:
            dataset["path"] = os.path.join(path_dict[dataset["dataset_name"]], dataset["path"])
            transforms = dataset.get("transforms", None)
            if transforms is not None:
                dataset["transforms"] = os.path.join(path_dict[dataset["dataset_name"]], dataset["transforms"])
    with open("/data/config/_datasets.json", "w") as f:
        json.dump(cfg, f, indent=4)
    return


if __name__ == '__main__':
    args = sys.argv
    update_dataset_config(args[1])
