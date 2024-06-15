#!/bin/bash
#SBATCH --mem-per-cpu 75g
#SBATCH -t 06:00:00
#SBATCH -o /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/logs/minmax_1600_sex.out
#SBATCH -e /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/logs/minmax_1600_sex.err
#SBATCH -J minmax_1600_sex
#SBATCH -A MomaBioinf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


source ~/.bashrc
conda activate /home/dmartinpestana/miniforge3/envs/optuna_GENIUS/


cd /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/src/image_maker || exit

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
