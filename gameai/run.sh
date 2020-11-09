#!/bin/sh
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_vmem=8G
#$ -l h_rt=6:0:0

module load singularity/3.6.1
module load python/3.6.3
module load java/1.8.0_152-oracle
source ~/projects/gameai_malmo/malmo/venv/bin/activate

python main.py
