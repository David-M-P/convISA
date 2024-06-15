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

path_to_config = (
    "/home/davidmartin/scGENIUSagg/cnn-isa/trials/example_trial/config.json"
)

with open(path_to_config, "r") as jsonfile:
    config = json.load(jsonfile)

path_to_meta = (
    "/home/davidmartin/scGENIUSagg/cnn-isa/trials/example_trial/assets/metadata.csv"
)
metadata = pd.read_csv(path_to_meta, sep=",")
image_folder = "/home/davidmartin/scGENIUSagg/cnn-isa/trials/example_trial/results/images/data_images"
predictor_column = 0
response_column = 2
path_to_trial = "/home/davidmartin/scGENIUSagg/cnn-isa/trials/example_trial"


dataset_f = OneK1KImageLoader(
    metadata,
    image_folder,
    predictor_column,
    response_column,
    filter_by_category=int(0),  # 1 male 0 female
)

number_of_bins = dataset_f.number_of_agebins

train_optimized(config, dataset_f, "female", path_to_trial, number_of_bins)
