# convISA

Welcome to convISA! A project in Bioinformatics done by David Martín Pestaña. This README file will contain all the information needed for the data format as well as to run the framework.

## Format of example_data directories

Feeding data to the neural network can be done only in one format, but the way that we preprocess data previous to this can hugely impact the rest of the process. One idea to use these example datasets is to copy the three csv files into a directory trials/TRIAL_NAME/assets. After that, if a config file is present in the base trial directory, the program can be run by using the single bash command with the selected trial name.

For each subdirectory in example_data, there will be a folder "scripts" that holds the python script used to create the data, as well as 3 files:

- metadata.csv: Will contain 3 columns and it is used as an index for the donor ids and their corresponding sex and agebin. The structure is:

    |donor_id (str)    | sex (int (F = 0, M = 1)) | age_bins (int)     |
    |:---:        |    :----:   |          :---: |
    |donor_1      | 0     | 3  |
    |donor_N   | 1       | 0     |

- expression.csv: Will contain as many columns as donors plus 1, it is the main container of the average expression data for each gene and cell type combination. The structure is:

    |donor_1 (float) |  donor_N (float)| gene_ctype (str) |
    |:---:           |          :---:  | :---:            |
    |expr value      | expr value      | gene1_celltype1  | 
    |expr value      | expr value      | geneN_celltypeN  |   

- gene_ctype.csv: Will serve as the dictionary to map the gene and cell type positions within the image created for each donor id. The structure is:

    |gene_ctype (str) |  gene_name (str)| cell_type (str) |c_type_num (int)|gene_num (int)|
    |:---:            |          :---:  | :---:            |:---:|:---:|
    |gene1_celltype1  | name of gene 1      | name of celltype 1  | index of cell type 1| index of gene 1|
    |geneN_celltypeN  | name of gene N      | name of celltype N  | index of cell type N| index of gene N|

## Index of directories

The following index is built in order of creation of datasets during the duration of the project:

> Note before: One observation means one expression value per cell type, gene and patient. After a certain step the table is pivoted and instead of observations, genes in donor+cell combination are filtered. 

A thorough explanation of each dataset can be seen below, for a shortened version, an overview of each preprocessing workflow can be seen in the following table.

<table align="center">
    <thead>
        <tr>
            <th style="text-align:center;"><b>csv file</b></th>
            <th style="text-align:center;"><b>Step</b></th>
            <th style="text-align:center;"><b>Index: 1</b></th>
            <th style="text-align:center;"><b>Index: 2</b></th>
            <th style="text-align:center;"><b>Index: 3</b></th>
            <th style="text-align:center;"><b>Index: 4</b></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align:center;"><b>Name</b></td>
            <td style="text-align:center;">Name</td>
            <td style="text-align:center;">Name</td>
            <td style="text-align:center;">Name</td>
            <td style="text-align:center;">Name</td>
            <td style="text-align:center;">Name</td>
        </tr>
        <tr>
            <td rowspan=4 style="text-align:center;"><b>Expression</b></td>
            <td style="text-align:center;">Type Prediction Thr</td>
            <td style="text-align:center;">50%</td>
            <td style="text-align:center;">50%</td>
            <td style="text-align:center;">50%</td>
            <td style="text-align:center;">50%</td>
        </tr>
        <tr>
            <td style="text-align:center;">Missing thr</td>
            <td style="text-align:center;">50%</td>
            <td style="text-align:center;">40%</td>
            <td style="text-align:center;">40%</td>            
            <td style="text-align:center;">40%</td>
        </tr>
        <tr>
            <td style="text-align:center;">Cell number</td>
            <td style="text-align:center;">19</td>
            <td style="text-align:center;">19</td>
            <td style="text-align:center;">19</td>
            <td style="text-align:center;">19</td>
        </tr>
        <tr>
            <td style="text-align:center;">Gene number</td>
            <td style="text-align:center;">295</td>
            <td style="text-align:center;">295</td>
            <td style="text-align:center;">295</td>
            <td style="text-align:center;">295</td>
        </tr>
        <tr>
            <td rowspan=2 style="text-align:center;"><b>Order</b></td>
            <td style="text-align:center;">Cells by</td>
            <td style="text-align:center;">Common + Pylogeny</td>
            <td style="text-align:center;">Common + Pylogeny</td>
            <td style="text-align:center;">Common + Pylogeny</td>
            <td style="text-align:center;">Common + Pylogeny</td>
        </tr>
        <tr>
            <td style="text-align:center;">Gene by</td>
            <td style="text-align:center;">Variable + STRING</td>
            <td style="text-align:center;">Variable + STRING</td>
            <td style="text-align:center;">Variable + STRING</td>
            <td style="text-align:center;">Variable + STRING</td>
        </tr>
        <tr>
            <td rowspan=2 style="text-align:center;"><b>Metadata</b></td>
            <td style="text-align:center;">Data</td>
            <td style="text-align:center;">Average per gene+ctype</td>
            <td style="text-align:center;">Average per gene+ctype</td>
            <td style="text-align:center;">Average per gene+ctype</td>
            <td style="text-align:center;">Average per gene+ctype</td>
        </tr>
        <tr>
            <td style="text-align:center;">Resampling</td>
            <td style="text-align:center;">No</td>
            <td style="text-align:center;">No</td>
            <td style="text-align:center;">Sex independent to 200</td>
            <td style="text-align:center;">Sex dependent to 200</td>
        </tr>
    </tbody>
</table>


## Thorough description of each dataset directory

1-onek1k_raw_noresample_50pcmiss: first dataset in which the model was trained, Anndata from OneK1K cohort was downloaded and parsed through the preprocessing script.

- Main processing steps for expression data:

    - Cell type prediction confidence threshold: 50%

        |Observations before| Filtered|Observations after|
        |:---:           |          :---:  |        :---:  |
        |1,248,980      | 193,210      |1,055,770     |

    - Selection of top 19 most common cells:

        |Observations before| Filtered|Observations after|
        |:---:           |          :---:  |        :---:  |
        |1,055,770      | 3,535      |1,052,235     |

    - Pivoting of table, filtering of non-coding genes:

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |36,571      | 17,599      |18,972     |

    - Threshold of missing values: 50%

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |18,972      | 13,812      |5,160     |

    - Extraction of top 295 most variable genes:

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |5,160      | 4,865      |295     |

- Main processing steps for gene and cell type ordering:

    - Cells ordered by phylogeny, genes ordered by STRING interaction representation.

- Main processing steps for metadata:

    - Sex was binary coded and agebins were created 0 to 3 with the following criteria: agebin0 $\le$ 54 | 55 $\le$ agebin1 $\le$ 64 | 65 $\le$ agebin2 $\le$ 74 | 75 $\le$ agebin3

2-onek1k_raw_noresample_40pcmiss: this dataset is similar to number 1 in essence, however a filter slighly more restrictive was used for missing values.

- Main processing steps for expression data:

    - Cell type prediction confidence threshold: 50%

        |Observations before| Filtered|Observations after|
        |:---:           |          :---:  |        :---:  |
        |1,248,980      | 193,210      |1,055,770     |

    - Selection of top 19 most common cells:

        |Observations before| Filtered|Observations after|
        |:---:           |          :---:  |        :---:  |
        |1,055,770      | 3,535      |1,052,235     |

    - Pivoting of table, filtering of non-coding genes:

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |36,571      | 17,599      |18,972     |

    - Threshold of missing values: 40%

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |18,972      | 15,399      |3,573     |

    - Extraction of top 295 most variable genes:

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |3,573      | 3,278      |295     |

- Main processing steps for gene and cell type ordering:

    - Cells ordered by phylogeny, genes ordered by STRING interaction representation.

- Main processing steps for metadata:

    - Sex was binary coded and agebins were created 0 to 3 with the following criteria: agebin0 $\le$ 54 | 55 $\le$ agebin1 $\le$ 64 | 65 $\le$ agebin2 $\le$ 74 | 75 $\le$ agebin3

3-onek1k_raw_nosexresample_40pcmiss: preprocessing of the data was done similarly to number 2. However, a resampling with replacement method was used in order to bump each agebin to 200 samples (independent of sex). The replacement was done with 50% of each sex for whole individuals not to break possible relationship between expression values.

- Main processing steps for expression data:

    - Cell type prediction confidence threshold: 50%

        |Observations before| Filtered|Observations after|
        |:---:           |          :---:  |        :---:  |
        |1,248,980      | 193,210      |1,055,770     |

    - Selection of top 19 most common cells:

        |Observations before| Filtered|Observations after|
        |:---:           |          :---:  |        :---:  |
        |1,055,770      | 3,535      |1,052,235     |

    - Pivoting of table, filtering of non-coding genes:

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |36,571      | 17,599      |18,972     |

    - Threshold of missing values: 40%

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |18,972      | 15,399      |3,573     |

    - Extraction of top 295 most variable genes:

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |3,573      | 3,278      |295     |

- Main processing steps for gene and cell type ordering:

    - Cells ordered by phylogeny, genes ordered by STRING interaction representation.

- Main processing steps for metadata:

    - Sex was binary coded and agebins were created 0 to 3 with the following criteria: agebin0 $\le$ 54 | 55 $\le$ agebin1 $\le$ 64 | 65 $\le$ agebin2 $\le$ 74 | 75 $\le$ agebin3
    - Resampling was done with replacement to get, at least 200 samples of each agebin (independently of sex, 200 total).

4- onek1k_raw_sexresample_40pcmiss: data similar to number 3. However, during resampling with replacement, independency of sex was taken, so each sex AND agebin are resampled to 200 samples. The replacement was done for whole individuals not to break possible relationship between expression values.

- Main processing steps for expression data:

    - Cell type prediction confidence threshold: 50%

        |Observations before| Filtered|Observations after|
        |:---:           |          :---:  |        :---:  |
        |1,248,980      | 193,210      |1,055,770     |

    - Selection of top 19 most common cells:

        |Observations before| Filtered|Observations after|
        |:---:           |          :---:  |        :---:  |
        |1,055,770      | 3,535      |1,052,235     |

    - Pivoting of table, filtering of non-coding genes:

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |36,571      | 17,599      |18,972     |

    - Threshold of missing values: 40%

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |18,972      | 15,399      |3,573     |

    - Extraction of top 295 most variable genes:

        |Genes before| Filtered|Genes after|
        |:---:           |          :---:  |        :---:  |
        |3,573      | 3,278      |295     |

- Main processing steps for gene and cell type ordering:

    - Cells ordered by phylogeny, genes ordered by STRING interaction representation.

- Main processing steps for metadata:

    - Sex was binary coded and agebins were created 0 to 3 with the following criteria: agebin0 $\le$ 54 | 55 $\le$ agebin1 $\le$ 64 | 65 $\le$ agebin2 $\le$ 74 | 75 $\le$ agebin3
    - Resampling was done with replacement to get, at least 200 samples of each agebin and sex.