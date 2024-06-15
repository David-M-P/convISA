import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))

import argparse
import json
import torch
import captum

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from functions_analysis import check_trial_dir, get_attr
from training.cnn_models import AE_flexible
from training.dataloader import OneK1KImageLoader
from image_maker.functions_images import make_image

parser = argparse.ArgumentParser(
    description="This script trains the network based on the settings selected. Provide a trial name that has a subdirectory in trials directory with a config file, it should also contain at least the images in a subdirectory 'results/images/data_images' as well as the best performing model as a .pb file."
)
parser.add_argument(
    "trial_name",
    type=str,
    help="Non-used trial name that has a subdirectory in trials directory with a config file, it should also contain at least the images in a subdirectory 'results/images/data_images' as well as the best performing model as a .pb file.",
)

args = parser.parse_args()

(
    current_directory,
    path_to_trial,
    path_to_config,
    path_to_meta,
    path_to_expr,
    path_to_order,
    path_to_images,
    path_to_model_males,
    path_to_model_females,
) = check_trial_dir(args.trial_name)

print(path_to_model_females)

with open(path_to_config, "r") as jsonfile:
    config = json.load(jsonfile)
    print("Read successful")


image_folder = path_to_images
metadata = pd.read_csv(path_to_meta, sep=",")
gene_ctype = pd.read_csv(path_to_order)


if config["train_for_age/sex"] == "age":
    predictor_column = 0
    response_column = 2
    if config["separate_males_females_yes/no"] == "yes":
        os.makedirs(os.path.join(path_to_trial, "results", "attribution", "female"))
        os.makedirs(os.path.join(path_to_trial, "results", "attribution", "male"))
        os.makedirs(os.path.join(path_to_trial, "results", "umap", "female"))
        os.makedirs(os.path.join(path_to_trial, "results", "umap", "male"))
        dataset_f = OneK1KImageLoader(
            metadata,
            image_folder,
            predictor_column,
            response_column,
            filter_by_category=0,  # 1 male 0 female
        )
        dataset_m = OneK1KImageLoader(
            metadata,
            image_folder,
            predictor_column,
            response_column,
            filter_by_category=1,  # 1 male 0 female
        )

        number_of_bins = dataset_f.number_of_agebins
        get_attr(
            config,
            dataset_f,
            "female",
            path_to_trial,
            number_of_bins,
            path_to_model_females,
            gene_ctype,
            "age",
        )
        get_attr(
            config,
            dataset_m,
            "male",
            path_to_trial,
            number_of_bins,
            path_to_model_males,
            gene_ctype,
            "age",
        )
    elif config["separate_males_females_yes/no"] == "no":
        os.makedirs(
            os.path.join(path_to_trial, "results", "attribution", "male_female")
        )
        os.makedirs(os.path.join(path_to_trial, "results", "umap", "male_female"))
        path_to_model_mix = path_to_model_males
        dataset = OneK1KImageLoader(
            metadata,
            image_folder,
            predictor_column,
            response_column,
            filter_by_category=None,  # 1 male 0 female
        )
        number_of_bins = dataset.number_of_agebins

        get_attr(
            config,
            dataset,
            "male_female",
            path_to_trial,
            number_of_bins,
            path_to_model_mix,
            gene_ctype,
            "age",
        )

elif config["train_for_age/sex"] == "sex":
    os.makedirs(os.path.join(path_to_trial, "results", "attribution", "male_female"))
    os.makedirs(os.path.join(path_to_trial, "results", "umap", "male_female"))
    predictor_column = 0
    response_column = 1
    path_to_model_mix = path_to_model_males
    dataset = OneK1KImageLoader(
        metadata,
        image_folder,
        predictor_column,
        response_column,
        filter_by_category=None,  # 1 male 0 female
    )
    number_of_bins = 2

    get_attr(
        config,
        dataset,
        "male_female",
        path_to_trial,
        number_of_bins,
        path_to_model_mix,
        gene_ctype,
        "sex",
    )
