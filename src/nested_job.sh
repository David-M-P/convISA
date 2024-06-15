#!/bin/bash
#SBATCH --mem-per-cpu 100g
#SBATCH --time=48:00:00
#SBATCH -o /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/logs/nested_job_newnet.out
#SBATCH -e /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/logs/nested_job_newnet.err
#SBATCH -J cnn_dmartinpestana_2
#SBATCH -A MomaBioinf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


source ~/.bashrc
conda activate /home/dmartinpestana/miniforge3/envs/optuna_GENIUS/


cd /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/src/ || exit

start_time=$(date +%s)

bash run_framework.sh 1600_sex_nosep_flex

end_time=$(date +%s)

total_time=$((end_time - start_time))

minutes=$((total_time / 60))

echo "Analysis 1 done in $minutes minutes"

cd /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/src/ || exit

start_time=$(date +%s)

bash run_framework.sh 1600_age_sep_flex

end_time=$(date +%s)

total_time=$((end_time - start_time))

minutes=$((total_time / 60))

echo "Analysis 2 done in $minutes minutes"

cd /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/src/ || exit

start_time=$(date +%s)

bash run_framework.sh 1600_age_nosep_flex

end_time=$(date +%s)

total_time=$((end_time - start_time))

minutes=$((total_time / 60))

echo "Analysis 3 done in $minutes minutes"

cd /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/src/ || exit

start_time=$(date +%s)

bash run_framework.sh 981_sex_nosep_flex

end_time=$(date +%s)

total_time=$((end_time - start_time))

minutes=$((total_time / 60))

echo "Analysis 4 done in $minutes minutes"

cd /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/src/ || exit

start_time=$(date +%s)

bash run_framework.sh 981_age_sep_flex

end_time=$(date +%s)

total_time=$((end_time - start_time))

minutes=$((total_time / 60))

echo "Analysis 5 done in $minutes minutes"

cd /faststorage/project/MomaBioinf/Public/scRNAseq/DerivedData/David2024/cnn_isa/src/ || exit

start_time=$(date +%s)

bash run_framework.sh 981_age_nosep_flex

end_time=$(date +%s)

total_time=$((end_time - start_time))

minutes=$((total_time / 60))

echo "Analysis 6 done in $minutes minutes"