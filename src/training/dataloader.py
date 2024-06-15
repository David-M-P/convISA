import glob
import pickle
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class OneK1KImageLoader(Dataset):
    """
    Data loader class, this will be specific to the format stated of metadata,
    expression and gene numbering, making it ideal for loading our preprocessed
    OneK1K data. However, it will work with data in same format which has been
    previously normalized.

    Takes as arguments the metadata (path to metadata), the folder where the
    expression images are, as well as the columns for predictor and response
    and option to filter by category. This last perk can be useful when training
    separately for each sex.
    """

    def __init__(
        self,
        metadata,
        folder,
        predictor_column,
        response_column,
        filter_by_category=None,
    ):

        self.annotation = metadata
        if filter_by_category is not None:
            self.annotation = self.annotation[
                self.annotation["sex"] == filter_by_category
            ]
        elif filter_by_category is None:
            self.number_of_sexes = len(self.annotation["sex"].unique())
        self.number_of_agebins = len(self.annotation["age_bins"].unique())
        self.folder = folder
        self.predictor_column = predictor_column
        self.response_column = response_column  # 1 for sex 2 for age_bins
        self.remove_rows_where_there_is_no_file()

    def remove_rows_where_there_is_no_file(self):
        """
        Function that removes the ids for the samples that have name in metadata but
        no image in folder.
        """
        print("Number of samples in metadata file: ", np.shape(self.annotation)[0])
        images = glob.glob("{}/*.dat".format(self.folder))
        ids = [os.path.basename(f).split(".")[0] for f in images]
        self.annotation = self.annotation[self.annotation["donor_id"].isin(ids)]
        print(
            "Number of images after removing missing files: ",
            np.shape(self.annotation)[0],
        )

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        """
        Main function of the dataloader, gets the image for each donor (x) as well
        as the response in order to feed it into the CNN.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open(
            "{}/{}.dat".format(
                self.folder, self.annotation.iloc[idx, self.predictor_column]
            ),
            "rb",
        ) as f:
            x = pickle.load(f)
            f.close()
        y = np.array(self.annotation.iloc[idx, self.response_column], dtype="long")

        return x, y, self.annotation.iloc[idx, self.predictor_column]
