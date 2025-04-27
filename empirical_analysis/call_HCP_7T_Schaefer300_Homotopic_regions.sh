#!/usr/bin/env bash

# Define the batch job array command
export github_dir=/taiji1/abry4213/github/info_theory_visuals/
export input_model_file=$github_dir/empirical_analysis/HCP_7T_fMRI_subject_run_list.txt

##################################################################################################
# Running pyspi across subjects -- homotopic connectivity
##################################################################################################

# Univariate
# python3 HCP_7T_Schaefer300_Homotopic_univariate_infotheory.py

# Pairwise
cmd="qsub -o $github_dir/cluster_output/HCP_7T_fMRI_infotheory_pyspi_^array_index^.out \
   -J 1-80 \
   -N HCP_7T_fMRI_pyspi \
   -l select=1:ncpus=1:mem=10GB:mpiprocs=1 \
   -v input_model_file=$input_model_file \
   call_HCP_7T_Schaefer300_Homotopic_regions.pbs"
$cmd