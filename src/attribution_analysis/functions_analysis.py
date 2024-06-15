import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))

import json
import torch
import captum
import umap

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from training.cnn_models import AE_flexible
from training.dataloader import OneK1KImageLoader
from image_maker.functions_images import make_image


def check_trial_dir(trial_name):
    """
    Function that takes as arguments the trial name and checks if it is a valid directory during
    the analysis phase of the CNN.
    """
    current_directory = os.getcwd()
    path_to_trial = os.path.join(current_directory, "..", "..", "trials", trial_name)
    path_to_config = os.path.join(path_to_trial, "config.json")
    path_to_meta = os.path.join(path_to_trial, "assets", "metadata.csv")
    path_to_expr = os.path.join(path_to_trial, "assets", "expression.csv")
    path_to_order = os.path.join(path_to_trial, "assets", "gene_ctype_order.csv")
    path_to_images = os.path.join(path_to_trial, "results", "images", "data_images")

    print("Checking if the trial directory is valid")
    if not os.path.exists(path_to_trial):
        raise FileNotFoundError(
            f"The trial name {trial_name} provided does not have a dedicated directory."
        )
    if not os.path.exists(path_to_config):
        raise FileNotFoundError(
            f"The directory {trial_name} does not have a dedicated config file."
        )
    if not os.path.exists(path_to_meta):
        raise FileNotFoundError(
            f"The directory {os.path.join(path_to_trial, 'assets')} does not have a metadata.csv file."
        )
    if not os.path.exists(path_to_expr):
        raise FileNotFoundError(
            f"The directory {os.path.join(path_to_trial, 'assets')} does not have a expression.csv file."
        )
    if not os.path.exists(path_to_order):
        raise FileNotFoundError(
            f"The directory {os.path.join(path_to_trial, 'assets')} does not have a gene_ctype_order.csv file."
        )
    if not os.path.exists(path_to_images):
        raise FileNotFoundError(
            f"The directory {path_to_images} does not exist or is not located inside the directory {trial_name}, results, images."
        )

    with open(path_to_config, "r") as jsonfile:
        config = json.load(jsonfile)
    if config["separate_males_females_yes/no"] == "no":
        path_to_model = os.path.join(
            path_to_trial,
            "results",
            "training",
            "male_female",
            "best_model",
            "best_model.pb",
        )
        if not os.path.exists(path_to_model):
            raise FileNotFoundError(
                f"The directory {path_to_model} does not exist or is not located inside the corresponding directory in {trial_name}."
            )
        else:
            return (
                current_directory,
                path_to_trial,
                path_to_config,
                path_to_meta,
                path_to_expr,
                path_to_order,
                path_to_images,
                path_to_model,
                None,
            )
    elif config["separate_males_females_yes/no"] == "yes":
        path_to_model_males = os.path.join(
            path_to_trial, "results", "training", "male", "best_model", "best_model.pb"
        )
        path_to_model_females = os.path.join(
            path_to_trial,
            "results",
            "training",
            "female",
            "best_model",
            "best_model.pb",
        )
        if not os.path.exists(path_to_model_males):
            raise FileNotFoundError(
                f"The directory {path_to_model_males} does not exist or is not located inside the corresponding directory in {trial_name}."
            )
        if not os.path.exists(path_to_model_females):
            raise FileNotFoundError(
                f"The directory {path_to_model_females} does not exist or is not located inside the corresponding directory in {trial_name}."
            )
        else:
            return (
                current_directory,
                path_to_trial,
                path_to_config,
                path_to_meta,
                path_to_expr,
                path_to_order,
                path_to_images,
                path_to_model_males,
                path_to_model_females,
            )


def create_UMAP(path_to_trial, sex, latent_shape):
    latent_1024_and_y = pd.read_csv(
        os.path.join(path_to_trial, "results", "umap", sex, "latent_vectors.csv"),
        header=None,
    )
    features = latent_1024_and_y.iloc[:, :latent_shape]
    y_dat = latent_1024_and_y.iloc[:, latent_shape:]
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(features)
    num_categories = len(y_dat.iloc[:, 0].unique())
    colors = plt.cm.viridis(
        np.linspace(0, 1, num_categories)
    )
    cmap = plt.cm.colors.ListedColormap(colors)
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_dat, cmap=cmap, s=5)
    plt.gca().set_aspect("equal", "datalim")
    colorbar_ax = plt.gcf().add_axes([0.95, 0.1, 0.03, 0.8])
    colorbar = plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap), cax=colorbar_ax)
    plt.savefig(
        os.path.join(path_to_trial, "results", "umap", sex, "umap.png"),
        bbox_inches="tight",
    )


def get_attr(
    config,
    dataset,
    sex,
    path_to_trial,
    number_of_bins,
    path_to_model,
    gene_ctype,
    age_or_sex,
):
    attribution_n_steps = config["attribution_n_steps"]
    if config["hyperparameter_optimization_yes/no"] == "yes":
        best_hyperparameters = pd.read_csv(
            os.path.join(
                path_to_trial,
                "results",
                "training",
                sex,
                "hyperparameter_training",
                f"optimized_hyperparameters_{sex}.csv",
            )
        )
        lr = float(best_hyperparameters["params_lr"][0])
        lr_decay = float(best_hyperparameters["params_lr_decay"][0])
        optimizer_name = best_hyperparameters["params_optimizer"][0]
        p_drop_ext = float(best_hyperparameters["params_p_drop_ext"][0])
        p_drop_pred = float(best_hyperparameters["params_p_drop_pred"][0])
        weight_decay = float(best_hyperparameters["params_weight_decay"][0])
        scale1 = int(best_hyperparameters["params_scale1"][0])
        scale2 = int(best_hyperparameters["params_scale2"][0])
        scale3 = int(best_hyperparameters["params_scale3"][0])
        scale4 = int(best_hyperparameters["params_scale4"][0])
        ext_scale = int(best_hyperparameters["params_ext_scale"][0])

    elif config["hyperparameter_optimization_yes/no"] == "no":
        lr = config["lr"]
        lr_decay = config["lr_decay"]
        optimizer_name = config["optimizer"]
        p_drop_ext = config["p_drop_ext"]
        p_drop_pred = config["p_drop_pred"]
        weight_decay = config["params_weight_decay"]
        scale1 = config["fixed_scale1"]
        scale2 = config["fixed_scale2"]
        scale3 = config["fixed_scale3"]
        scale4 = config["fixed_scale4"]
        ext_scale = config["fixed_ext_scale"]

    net = AE_flexible(p_drop_ext, p_drop_pred, scale1, scale2, scale3, scale4,ext_scale, number_of_bins)
    checkpoint = torch.load(path_to_model)
    net.load_state_dict(checkpoint["model_state_dict"])

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(
            net.parameters(), lr_decay=lr_decay, lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def wrapped_model(inp):
        return net(inp)[0]

    def wrapped_model_L(inp):
        return net(inp)[2]

    net.eval()
    occlusion = captum.attr.IntegratedGradients(wrapped_model)
    latent_shape = scale1*scale2*scale3*scale4
    if dataset.annotation.shape[0] != 0:
        trainLoader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
        print(f"Total Samples for {sex}: ", len(trainLoader))
        heatmaps_exp = []
        latent_vectors = np.zeros((0, latent_shape+1))
        for x, y_dat, id in tqdm(trainLoader):
            latent = wrapped_model_L(x)
            latent = latent.detach().numpy()
            latent_with_y = np.zeros(latent_shape+1)
            latent_with_y[:latent_shape] = latent
            latent_with_y[latent_shape] = y_dat
            latent_vectors = np.vstack((latent_vectors, latent_with_y))
            if y_dat == 1:
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)
                baseline = torch.zeros((1, x.shape[1], x.shape[2], x.shape[3]))
                attribution = occlusion.attribute(
                    x, baseline, target=1, n_steps=int(attribution_n_steps)
                )
                attribution = attribution.squeeze(0).cpu().detach().numpy()
                heatmaps_exp.append(np.abs(attribution[0, :, :]))

        np.savetxt(
            os.path.join(path_to_trial, "results", "umap", sex, "latent_vectors.csv"),
            latent_vectors,
            delimiter=",",
        )

        create_UMAP(path_to_trial, sex, latent_shape)

        heatmaps_exp = np.array(heatmaps_exp)
        mean_exp_matrix = heatmaps_exp.mean(axis=0)
        plt.clf()
        fig, ax = plt.subplots()
        ax = sns.heatmap(mean_exp_matrix, cmap="YlGnBu")
        plt.title(f"Attribution score for {age_or_sex} prediction of {sex}")
        plt.savefig(
            os.path.join(
                path_to_trial, "results", "attribution", sex, "attribution_matrix.png"
            ),
            dpi=200,
        )

        number_of_genes_returned = gene_ctype.shape[0] - 1

        image = make_image("ID", 1, 1, gene_ctype)
        exp_att = image.analyze_attribution(mean_exp_matrix, number_of_genes_returned)
        exp_att.columns = ["attribution", "gene_ctype"]
        exp_att.reset_index(drop=True, inplace=True)
        exp_att.index += 1
        total_df = pd.concat([exp_att])

        total_df.to_csv(
            os.path.join(
                path_to_trial, "results", "attribution", sex, "attribution_matrix.csv"
            )
        )
        print(
            f"Attribution scores are saved in {os.path.join(path_to_trial, 'results', 'attribution', sex, 'attribution_matrix.csv')}.csv file."
        )
