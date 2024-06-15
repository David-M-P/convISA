import itertools
import numpy as np
import pandas as pd


class ImagePixel:
    """
    Class containing the important attributes for each pixel of each image. Contains the gene and cell type as
    well as the i and j coordinates within the image and the expression value of the gene in the corresponding cell type.
    """

    def __init__(self, gene, exp_val, cell_type):
        self.gene = gene
        self.exp_val = exp_val
        self.cell = cell_type
        self.i = -1
        self.j = -1


class Image:
    """
    Class containing the important attributes for each image. There will be one image per sample. Also contains
    two functions, make_matrix relevant when building the images, and analyze_attribution, used when studying the
    relative importance of each combination of gene and cell type for the prediction.
    """

    def __init__(self, donor_id, pixel_dict, sex, age):
        self.id = donor_id
        self.pixel_dict = pixel_dict
        self.sex = sex
        self.age = age

    def make_matrix(self, shape):
        """
        Takes as argument a tuple with format (rows, columns) for the image that will be made.
        Iterates through each gene and cell type combination for each donor and places in the corresponding
        pixel of the image (with coordinates i and j) the expression value.
        In case that no expression is seen for a gene and cell type combination, it is taken as 0.
        Returns a numpy array of said shape in which each i and j point corresponds to an expression value.
        """
        n_cells = shape[0]
        n_genes = shape[1]
        exp = np.zeros((n_cells, n_genes))
        for gene_ctype in self.pixel_dict:
            if self.pixel_dict[gene_ctype].exp_val is not None:
                exp[self.pixel_dict[gene_ctype].i, self.pixel_dict[gene_ctype].j] = (
                    self.pixel_dict[gene_ctype].exp_val
                )
        exp = np.nan_to_num(exp, nan=0)
        image = np.asarray(exp, dtype="float32")
        return image

    def analyze_attribution(self, att_mat, n):
        """
        Takes as argument the attribution matrix, a numpy array that shows the relative attribution of each
        gene and cell type and the number of genes and cell types for which the attribution should be shown.
        For each combination of gene and cell type, sorts the attribution scores in descending order and puts
        the n top genes in a pandas dataframe.
        Returns the attribution score dataframe.
        """
        attr_dict = {}
        for gene_ctype in self.pixel_dict:
            attr_dict[gene_ctype] = att_mat[
                self.pixel_dict[gene_ctype].i, self.pixel_dict[gene_ctype].j
            ]
        sort_dict = {
            k: v
            for k, v in sorted(
                attr_dict.items(), key=lambda item: item[1], reverse=True
            )
        }
        sort_dict = dict(itertools.islice(sort_dict.items(), n))
        df = pd.DataFrame.from_dict(sort_dict, orient="index")
        df["gene_ctype"] = df.index
        df.reset_index(drop=True)
        return df
