# convISA

Welcome to convISA! A project in Bioinformatics done by David Martín Pestaña. This README file will contain all the information needed for the data format as well as to run the framework.
For the full project report PDF, it can be accessed in this same directory.

## Format of example_data directories

Feeding data to the neural network can be done only in one format, but the way that we preprocess data previous to this can hugely impact the rest of the process. The network runs as long as there is a directory with a particular name inside of "trials" (as by default only example_trial is there). The directory should have a "config.json" file and an "assets" subdirectory in which the three files needed to run the framework will be.

Each of the 3 files in assets directory should be:

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

> Note before: One observation means one expression value per cell type, gene and patient. After a certain step the table is pivoted and instead of observations, genes in donor+cell combination are filtered. 

## Description on how to run ConvISA.

In order to run ConvISA, place the full repository in an HPC cluster with SLURM queuing system, the conda environment can be directly installed from the environment.yaml file in the repository. Then, job_cluster.sh has to be modified in every place that says "REPLACE" (suggested to do Ctrl+Find in the file with "REPLACE" as 4 parameters are to be changed with the cluster username, project/job name and directory). Once done that, navigate to convisa/src and run the bash command "sbatch run_cluster.sh $trial_name" (trial_name should be a directory with a valid structure as mentioned before). If everything is valid a job will be sent for queue in a GPU node and it will be run.

## Thorough description of the steps followed in order to get each dataset

- Main processing steps for expression data

    - Cell type prediction confidence threshold: 50%


    - Selection of top 19 most common cells


    - Pivoting of table, filtering of non-coding genes


    - Threshold of missing values: 40%


    - Extraction of top 295 most variable genes


- Main processing steps for gene and cell type ordering:

    - Cells ordered by phylogeny, genes ordered by STRING interaction representation.

- Main processing steps for metadata

    - Sex was binary coded and agebins were created 0 to 3 with the following criteria: agebin0 $\le$ 54 | 55 $\le$ agebin1 $\le$ 64 | 65 $\le$ agebin2 $\le$ 74 | 75 $\le$ agebin3
    - Resampling was done with replacement to get, at least 200 samples of each agebin and sex.