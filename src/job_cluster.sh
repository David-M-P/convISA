#!/bin/bash
#SBATCH --mem-per-cpu 75g
#SBATCH -t 06:00:00
#SBATCH -o REPLACE_WITH_DIRECTORY/cnn_isa/logs/minmax_1600_sex.out
#SBATCH -e REPLACE_WITH_DIRECTORY/cnn_isa/logs/minmax_1600_sex.err
#SBATCH -J REPLACE_WITH_JOB_NAME
#SBATCH -A REPLACE_WITH_PROJECT_NAME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


source ~/.bashrc
conda activate /home/REPLACE_WITH_CLUSTER_USERNAME/miniforge3/envs/ConvISA/


cd REPLACE_WITH_DIRECTORY/cnn_isa/src/image_maker || exit

start_time=$(date +%s)

python run_image_maker.py "$1"

cd ../training || exit

python run_train.py "$1"

cd ../attribution_analysis || exit

python run_analysis.py "$1"

end_time=$(date +%s)

total_time=$((end_time - start_time))

minutes=$((total_time / 60))

echo "Analysis done in $minutes minutes"
