import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))

import pandas as pd
import numpy as np
import pickle
import time
import argparse
from tqdm import tqdm
from functions_images import make_image, find_expr, check_trial_dir

parser = argparse.ArgumentParser(
    description="This script creates the expression images, provide a non-used trial name that has a subdirectory in trials directory with a config file"
)
parser.add_argument(
    "trial_name",
    type=str,
    help="Non-used trial name that has a subdirectory in trials directory with a config file",
)

args = parser.parse_args()

print("Checking if the trial directory is valid")
(
    current_directory,
    path_to_trial,
    path_to_config,
    path_to_meta,
    path_to_expr,
    path_to_order,
) = check_trial_dir(args.trial_name)

print("Creating 'results/image' directory")

os.makedirs(os.path.join(path_to_trial, "results", "images", "data_images"))
os.makedirs(os.path.join(path_to_trial, "results", "images", "sample_images"))

print("Reading metadata, expression and gene_ctype order")
metadata = pd.read_csv(path_to_meta)
print("Reading expression")
expression = pd.read_csv(path_to_expr)
print("Reading gene_ctype_order")
gene_ctype = pd.read_csv(path_to_order)

print("Creating images")
start_time = time.time()
for index, row in tqdm(metadata.iterrows(), total=len(metadata)):

    donor_id = row["donor_id"]
    age = row["age_bins"]
    sex = row["sex"]

    image = make_image(donor_id, sex, age, gene_ctype)
    image = find_expr(donor_id, image, expression)
    final_exp_image = image.make_matrix((19, 295))

    sample_save_path = os.path.join(
        path_to_trial, "results", "images", "sample_images", f"{donor_id}.txt"
    )
    data_save_path = os.path.join(
        path_to_trial, "results", "images", "data_images", f"{donor_id}.dat"
    )

    np.savetxt(sample_save_path, final_exp_image, fmt="%f", delimiter="\t")

    with open(data_save_path, "wb") as f:
        pickle.dump(final_exp_image, f)
        f.close()


print("Done in --- %s minutes ---" % ((time.time() - start_time) / 60))
