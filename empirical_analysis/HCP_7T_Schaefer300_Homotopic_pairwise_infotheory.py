import pandas as pd
import numpy as np
from copy import deepcopy
from glob import glob
import os
import os.path as op
import random

# Statistics
from scipy.stats import zscore

# Java/pyspi
import jpype
from pyspi.calculator import Calculator
from pyspi.data import Data
import pickle

# Parallelization
from joblib import Parallel, delayed

# Set seed to 127
random.seed(127)

import argparse

###################################################################################################
parser=argparse.ArgumentParser()
parser.add_argument('--subject_run',
                    type=str,
                    default='100307_run1',
                    help='subject_ID plus run number (e.g. "100307_run1")')
parser.add_argument('--input_data_path',
                    type=str,
                    default='/taiji1/abry4213/data/HCP/HCP_7T/',
                    help='subject_ID plus run number (e.g. "100307_run1")')
parser.add_argument('--output_data_path',
                    type=str,
                    default='/taiji1/abry4213/github/info_theory_visuals/data/')
parser.add_argument('--SPI_subset',
                    type=str,
                    default='fast',
                    help='Subset of SPIs to compute')
opt=parser.parse_args()

# Parse arguments
subject_run = opt.subject_run
input_data_path = opt.input_data_path
output_data_path = opt.output_data_path
SPI_subset = opt.SPI_subset

subject_ID = subject_run.split("_")[0]
run_number = int(subject_run.split("_")[1].replace("run", ""))

pyspi_output_data_path = output_data_path + "pyspi_outputs/"

# If SPI_subset ends in .yaml, 
SPI_subset_base = op.basename(SPI_subset).replace(".yaml", "")

# Set seed to 127
random.seed(127)

def pairwise_info_theory_measures_for_parcellation_file(subject_TS, subject_ID, run_number,
                                                        calc, schaefer300_homotopic_lookup):
    """
    Process a single parcellation file to compute pairwise information-theoretic measures.
    Args:
        input_parc_file (str): Path to the parcellation file.
    Returns:
        pd.DataFrame: DataFrame containing entropy and AIS values for each region.    
    """

    ########## pyspi ##########

    # Load the data -- use both z-score normalization and linear detrending for fMRI
    calc.load_dataset(Data(subject_TS, normalise=True, detrend=True))
    calc.compute()
    this_measure_subject_calc_res = deepcopy(calc.table)

    # Iterate over each measure
    this_measure_subject_calc_res.columns = this_measure_subject_calc_res.columns.to_flat_index()

    # Convert index to column
    this_measure_subject_calc_res.reset_index(level=0, inplace=True)

    # Rename index as first brain region
    this_measure_subject_calc_res = this_measure_subject_calc_res.rename(columns={"index": "region_index_from"})

    # Pivot data from wide to long
    this_measure_subject_calc_res_long = pd.melt(this_measure_subject_calc_res, id_vars="region_index_from")

    # Separate the region index to + measure
    this_measure_subject_calc_res_long['Measure'] = this_measure_subject_calc_res_long['variable'].apply(lambda x: x[0])
    this_measure_subject_calc_res_long['region_index_to'] = this_measure_subject_calc_res_long['variable'].apply(lambda x: x[1])
    this_measure_subject_calc_res_long = (this_measure_subject_calc_res_long
                                    .drop(columns=['variable'])
                                    .query("region_index_from != region_index_to")
                                    .assign(region_index_from = lambda x: x.region_index_from.str.replace("proc-", ""),
                                            region_index_to = lambda x: x.region_index_to.str.replace("proc-", ""))
                                            .assign(region_index_from = lambda x: x.region_index_from.astype(int),
                                                    region_index_to = lambda x: x.region_index_to.astype(int))
                                    .merge(schaefer300_homotopic_lookup, left_on="region_index_from", right_on="Region_Index", how="left")
                                    .rename(columns={"Region": "Brain_Region_from"})
                                    .drop(columns=["Region_Index", "region_index_from"])
                                    .merge(schaefer300_homotopic_lookup, left_on="region_index_to", right_on="Region_Index", how="left")
                                    .rename(columns={"Region": "Brain_Region_to"})
                                    .drop(columns=["Region_Index", "region_index_to"])
                                    )
    
    # Add key information
    this_measure_subject_calc_res_long = (this_measure_subject_calc_res_long
                                      .assign(Subject_ID = subject_ID,
                                              Run_Number = run_number,
                                              Measure_Type = "Bivariate")
        
    )
    return this_measure_subject_calc_res_long

###################################################################################################

if __name__ == "__main__":
    
    this_subject_run_output_file = f'{pyspi_output_data_path}/{subject_ID}_run_{run_number}_HCP_7T_four_runs_Schaefer300_homotopic_pairwise_infotheory.csv'
    if not os.path.isfile(this_subject_run_output_file):

        print(f"Calculating pairwise information-theoretic measures for {subject_ID} run {run_number}...")

        # Initialise a base calculator
        if SPI_subset == "fast":
            calc = Calculator(subset='fast')
        else:
            calc = Calculator(configfile=SPI_subset)

        # Parse this subject's input file
        this_subject_run_input_file = f"{input_data_path}/{subject_ID}_rest7T_runs{run_number}__Y300.npy"
        this_subject_run_TS = np.load(this_subject_run_input_file).T

        # Define Schaefer-300 homotopic atlas lookup table (from Yan 2023)
        schaefer300_homotopic_lookup = pd.read_table(f'{output_data_path}/300Parcels_Yeo2011_7Networks_info.txt', header=None)

        # Take every other row of schaefer300_homotopic_lookup, ignoring the lookup color rows
        schaefer300_homotopic_lookup = (schaefer300_homotopic_lookup
                            .iloc[::2, :]
                            .reset_index(drop=True)
                            .rename(columns={0: 'Region'})
                            .assign(Region_Index = lambda x: range(x.shape[0]))
        )

        pairwise_infotheory_measure_results_HCP_7T_fMRI = pairwise_info_theory_measures_for_parcellation_file(subject_TS=this_subject_run_TS, 
                                                                                                              subject_ID=subject_ID, 
                                                                                                              run_number=run_number,
                                                                                                              calc=calc, 
                                                                                                              schaefer300_homotopic_lookup=schaefer300_homotopic_lookup)
        
        # Save to CSV
        pairwise_infotheory_measure_results_HCP_7T_fMRI.to_csv(this_subject_run_output_file, index=False)
