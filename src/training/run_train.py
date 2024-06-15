import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))


import json

import argparse
import random
import pandas as pd

from tqdm import tqdm
import optuna
from optuna.trial import TrialState

import numpy as np
import torch

from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataloader import OneK1KImageLoader
from cnn_models import AE_fixed, AE_flexible, AE_optuna
from functions_train import *
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(
    description="This script trains the network based on the settings selected. Provide a trial name that has a subdirectory in trials directory with a config file, it should also contain at least the images in a subdirectory 'results/images/data_images'"
)
parser.add_argument(
    "trial_name",
    type=str,
    help="Non-used trial name that has a subdirectory in trials directory with a config file, it should also contain at least the images in a subdirectory 'results/images/data_images'",
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
    path_to_images,
) = check_trial_dir(args.trial_name)

print("Reading settings")
with open(path_to_config, "r") as jsonfile:
    config = json.load(jsonfile)


if config["train_for_age/sex"] == "sex":
    if config["separate_males_females_yes/no"] == "yes":
        raise KeyError(
            "Error! You cannot separate training by sex when it is the response variable, check the configuration."
        )


image_folder = path_to_images
metadata = pd.read_csv(path_to_meta, sep=",")

print(
    f"Read successful, you will be training the model with the following general settings:\nPredict: {config['train_for_age/sex']}\nHyperparameter optimization: {config['hyperparameter_optimization_yes/no']}\nMale/female distinction at training: {config['separate_males_females_yes/no']}"
)

print("Creating 'results/training' directory")

os.makedirs(os.path.join(path_to_trial, "results", "training"))

if config["hyperparameter_optimization_yes/no"] == "yes":
    if config["train_for_age/sex"] == "age":
        predictor_column = 0
        response_column = 2
        if config["separate_males_females_yes/no"] == "yes":
            os.makedirs(
                os.path.join(
                    path_to_trial, "results", "training", "female", "loss_plot"
                )
            )
            os.makedirs(
                os.path.join(path_to_trial, "results", "training", "male", "loss_plot")
            )
            os.makedirs(
                os.path.join(
                    path_to_trial, "results", "training", "female", "best_model"
                )
            )
            os.makedirs(
                os.path.join(path_to_trial, "results", "training", "male", "best_model")
            )
            os.makedirs(
                os.path.join(
                    path_to_trial,
                    "results",
                    "training",
                    "female",
                    "hyperparameter_training",
                )
            )
            os.makedirs(
                os.path.join(
                    path_to_trial,
                    "results",
                    "training",
                    "male",
                    "hyperparameter_training",
                )
            )

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

            run_trial(config, dataset_f, "female", path_to_trial, number_of_bins)
            train_optimized(config, dataset_f, "female", path_to_trial, number_of_bins)

            run_trial(config, dataset_m, "male", path_to_trial, number_of_bins)
            train_optimized(config, dataset_m, "male", path_to_trial, number_of_bins)

        elif config["separate_males_females_yes/no"] == "no":
            os.makedirs(
                os.path.join(
                    path_to_trial, "results", "training", "male_female", "loss_plot"
                )
            )
            os.makedirs(
                os.path.join(
                    path_to_trial, "results", "training", "male_female", "best_model"
                )
            )
            os.makedirs(
                os.path.join(
                    path_to_trial,
                    "results",
                    "training",
                    "male_female",
                    "hyperparameter_training",
                )
            )

            dataset = OneK1KImageLoader(
                metadata,
                image_folder,
                predictor_column,
                response_column,
                filter_by_category=None,  # 1 male 0 female
            )
            number_of_bins = dataset.number_of_agebins
            run_trial(config, dataset, "male_female", path_to_trial, number_of_bins)
            train_optimized(
                config, dataset, "male_female", path_to_trial, number_of_bins
            )

    elif config["train_for_age/sex"] == "sex":
        predictor_column = 0
        response_column = 1
        os.makedirs(
            os.path.join(
                path_to_trial, "results", "training", "male_female", "loss_plot"
            )
        )
        os.makedirs(
            os.path.join(
                path_to_trial, "results", "training", "male_female", "best_model"
            )
        )
        os.makedirs(
            os.path.join(
                path_to_trial,
                "results",
                "training",
                "male_female",
                "hyperparameter_training",
            )
        )
        dataset = OneK1KImageLoader(
            metadata,
            image_folder,
            predictor_column,
            response_column,
            filter_by_category=None,  # 1 male 0 female
        )
        number_of_bins = 2

        run_trial(config, dataset, "male_female", path_to_trial, number_of_bins)
        train_optimized(config, dataset, "male_female", path_to_trial, number_of_bins)


elif config["hyperparameter_optimization_yes/no"] == "no":
    if config["train_for_age/sex"] == "age":
        predictor_column = 0
        response_column = 2
        if config["separate_males_females_yes/no"] == "yes":
            os.makedirs(
                os.path.join(
                    path_to_trial, "results", "training", "female", "loss_plot"
                )
            )
            os.makedirs(
                os.path.join(path_to_trial, "results", "training", "male", "loss_plot")
            )
            os.makedirs(
                os.path.join(
                    path_to_trial, "results", "training", "female", "best_model"
                )
            )
            os.makedirs(
                os.path.join(path_to_trial, "results", "training", "male", "best_model")
            )

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

            train_fixed(config, dataset_f, "female", path_to_trial, number_of_bins)
            train_fixed(config, dataset_m, "male", path_to_trial, number_of_bins)

        elif config["separate_males_females_yes/no"] == "no":
            os.makedirs(
                os.path.join(
                    path_to_trial, "results", "training", "male_female", "loss_plot"
                )
            )
            os.makedirs(
                os.path.join(
                    path_to_trial, "results", "training", "male_female", "best_model"
                )
            )

            dataset = OneK1KImageLoader(
                metadata,
                image_folder,
                predictor_column,
                response_column,
                filter_by_category=None,  # 1 male 0 female
            )
            number_of_bins = dataset.number_of_agebins
            train_fixed(config, dataset, "male_female", path_to_trial, number_of_bins)

    elif config["train_for_age/sex"] == "sex":
        predictor_column = 0
        response_column = 1
        os.makedirs(
            os.path.join(
                path_to_trial, "results", "training", "male_female", "loss_plot"
            )
        )
        os.makedirs(
            os.path.join(
                path_to_trial, "results", "training", "male_female", "best_model"
            )
        )

        dataset = OneK1KImageLoader(
            metadata,
            image_folder,
            predictor_column,
            response_column,
            filter_by_category=None,  # 1 male 0 female
        )
        number_of_bins = 2

        train_fixed(config, dataset, "male_female", path_to_trial, number_of_bins)
