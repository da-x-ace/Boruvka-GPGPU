#!/bin/bash

#$ -V
#$ -cwd
#$ -q gpu
#$ -pe 1way 12
#$ -N prob
#$ -o output_prob
#$ -e error_prob
#$ -M duke.lnmiit@gmail.com
#$ -m be
#$ -l h_rt=01:00:00

module load cuda
set -x
./prob > prob_output_100000_new
