#!/bin/bash
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --container-mounts=/data/diag:/data/diag
#SBATCH --container-image="dockerdex.umcn.nl:5005#rubenvdwaerden1997/train_monai:v1.4"
#SBATCH -o ./slurm_output/output/slurm_output_%j.txt
#SBATCH -e ./slurm_output/error/slurm_error_%j.txt
#SBATCH --exclude=dlc-electabuzz,dlc-scyther,dlc-mewtwo


sbatch /data/diag/Jlucas/install_packages.sh

python3 -u /data/diag/Jlucas/tools/stratified_split_trainvaltest.py
python3 -u /data/diag/Jlucas/tools/add_pathing.py
python3 -u /data/diag/Jlucas/tools/check_and_count_uniques.py


