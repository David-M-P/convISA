import numpy as np
import os
from static.classes import ImagePixel, Image


def check_trial_dir(trial_name):
    """
    Function that takes as arguments the trial name and checks if it is a valid directory during
    the image making phase of the CNN.
    """
    current_directory = os.getcwd()
    path_to_trial = os.path.join(current_directory, "..", "..", "trials", trial_name)
    path_to_config = os.path.join(path_to_trial, "config.json")
    path_to_meta = os.path.join(path_to_trial, "assets", "metadata.csv")
    path_to_expr = os.path.join(path_to_trial, "assets", "expression.csv")
    path_to_order = os.path.join(path_to_trial, "assets", "gene_ctype_order.csv")
    print("Checking if the trial directory is valid")
    if not os.path.exists(path_to_trial):
        raise FileNotFoundError(
            f"The trial name '{trial_name}' provided does not have a dedicated directory."
        )
    if not os.path.exists(path_to_config):
        raise FileNotFoundError(
            f"The directory '{trial_name}' does not have a dedicated config file."
        )
    if not os.path.exists(path_to_meta):
        raise FileNotFoundError(
            f"The directory '{os.path.join(path_to_trial, 'assets')}' does not have a metadata.csv file."
        )
    if not os.path.exists(path_to_expr):
        raise FileNotFoundError(
            f"The directory '{os.path.join(path_to_trial, 'assets')}' does not have a expression.csv file."
        )
    if not os.path.exists(path_to_order):
        raise FileNotFoundError(
            f"The directory '{os.path.join(path_to_trial, 'assets')}' does not have a gene_ctype_order.csv file."
        )
    else:
        return (
            current_directory,
            path_to_trial,
            path_to_config,
            path_to_meta,
            path_to_expr,
            path_to_order,
        )


def make_image(donor_id, sex, age, gene_celltype_index):
    """
    Function that takes as arguments the donor_id (str), sex (int), age (int) and gene_celltype
    index (pd.Dataframe). It assigns each pixel a i and j coordinate based on the corresponding
    gene and cell type combination. Then returns the Image class for the corresponding donor.
    """
    dictionary = {}
    for i, row in gene_celltype_index.iterrows():
        img = ImagePixel(row["gene_name"], exp_val=None, cell_type=row["cell_type"])
        img.i = row["c_type_num"]
        img.j = row["gene_num"]
        dictionary[row["gene_ctype"]] = img
    return Image(donor_id=donor_id, sex=sex, age=age, pixel_dict=dictionary)


def find_expr(donor_id, image, gene_exp):
    """
    Function that takes as arguments the donor_id (str), image (class Image) and gene_exp
    (pd.Dataframe). It iterates through each column in the expression array and poopulates
    each pixel of the image with the corresponding expression value for the gene and cell
    type combination.
    Returns the final image populated with the expression values.
    """
    if donor_id in np.array(gene_exp.columns):
        genes_ctypes = gene_exp["gene_ctype"]
        exp = gene_exp[donor_id]
        for i in range(len(genes_ctypes)):
            if genes_ctypes[i] in image.pixel_dict:
                image.pixel_dict[genes_ctypes[i]].exp_val = exp[i]
    return image
