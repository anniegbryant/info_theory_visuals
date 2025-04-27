import pandas as pd
import numpy as np
from copy import deepcopy
from glob import glob
import os
import random

# Statistics
from scipy.stats import zscore

# Java/pyspi
import jpype
from pyspi.calculator import Calculator
from pyspi.data import Data

# Parallelization
from joblib import Parallel, delayed

# Set seed to 127
random.seed(127)

# Set n_jobs
n_jobs = 4

###################################################################################################
github_base_path = "/taiji1/abry4213/github/"

# Point to pyspi installation of infodynamics
jarLocation = f"{github_base_path}/pyspi/pyspi/lib/jidt/infodynamics.jar"

# Check if a JVM has already been started
# If not, start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# Set seed to 127
random.seed(127)

# Create a Gaussian entropy calculator
entropy_calcClass = jpype.JPackage("infodynamics.measures.continuous.kozachenko").EntropyCalculatorMultiVariateKozachenko
entropy_calc = entropy_calcClass()
entropy_calc.initialise()

# Create a Gaussian Active Information Storage calculator
AIScalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
AIScalc = AIScalcClass()
AIScalc.initialise()

# Add observations
AIScalc.setProperty("NORMALISE", "true")

###################################################################################################
def info_theory_measures_for_parcellation_file(input_parc_file, basecalc, schaefer300_homotopic_lookup):
    """
    Process a single parcellation file to compute entropy and AIS for each region.
    Args:
        input_parc_file (str): Path to the parcellation file.
    Returns:
        pd.DataFrame: DataFrame containing entropy and AIS values for each region.    
    """
    this_file_base = os.path.basename(input_parc_file)

    # Get the subject ID from the file name
    subject_ID = this_file_base.split("_")[0]

    # run_number will be contained within e.g. '_runs1__', where we look for a number between 0 and 3, extract
    run_number = int(this_file_base.split("_runs")[1].split("__")[0])

    # Load the parcellation file, with regions as rows
    subject_TS = np.load(input_parc_file).T

    this_run_univariate_res_list = []

    for region_index in range(subject_TS.shape[0]):

        # Get the time series for this region
        region_TS = subject_TS[region_index, :]

        # z-score the time series
        region_TS = zscore(region_TS)

        # Convert to Java array
        region_TS_java = jpype.JArray(jpype.JDouble)([float(x) for x in region_TS])

        # Compute entropy for the source region
        region_entropy_calc = deepcopy(entropy_calc)
        region_entropy_calc.setObservations(region_TS_java)
        region_entropy = region_entropy_calc.computeAverageLocalOfObservations()

        # Compute AIS for the source region
        region_AIS_calc = deepcopy(AIScalc)
        region_AIS_calc.setObservations(region_TS_java)
        region_AIS = region_AIS_calc.computeAverageLocalOfObservations()

        # Compile dataframe
        this_region_univariate_res = pd.DataFrame({
            "Region_Index": [region_index, region_index],
            "Measure_Type": ["Univariate", "Univariate"],
            "Measure": ["Entropy", "AIS"],
            "Value": [region_entropy, region_AIS],
            "Subject_ID": [subject_ID, subject_ID],
            "Run_Number": [run_number, run_number],
        })

        # Append to the list
        this_run_univariate_res_list.append(this_region_univariate_res)

    # Concatenate the results
    this_run_univariate_res = pd.concat(this_run_univariate_res_list, ignore_index=True)

    ########## pyspi ##########
    # Make a copy of calc and compute
    this_measure_subject_calc = deepcopy(basecalc)

    # Load the data -- use both z-score normalization and linear detrending for fMRI
    this_measure_subject_calc.load_dataset(Data(subject_TS, normalise=True, detrend=True))
    this_measure_subject_calc.compute()
    this_measure_subject_calc_res = deepcopy(this_measure_subject_calc.table)

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

    # Concatenate and return the univariate + bivariate results
    this_run_res = pd.concat([this_run_univariate_res, this_measure_subject_calc_res_long], ignore_index=True)
    return this_run_res

###################################################################################################

if __name__ == "__main__":

    # Load measure groupings
    infotheory_measure_info = pd.read_csv(f"{github_base_path}/info_theory_visuals/data/infotheory_measure_info.csv")

    # Load in the HCP 7T fMRI data in Schaefer-300 (homotopic) parcellation space from Jayson
    HCP_7T_fMRI_path = "/taiji1/abry4213/data/HCP/HCP_7T/"
    parc_files = glob(f"{HCP_7T_fMRI_path}*Y300.npy")

    # Only proceed if there are parcellation files
    if len(parc_files) == 0:
        raise ValueError("No parcellation files found in the specified directory.")

    # Define Schaefer-300 homotopic atlas lookup table (from Yan 2023)
    schaefer300_homotopic_lookup = pd.read_table(f'{github_base_path}/info_theory_visuals/data/300Parcels_Yeo2011_7Networks_info.txt', header=None)

    # Take every other row of schaefer300_homotopic_lookup, ignoring the lookup color rows
    schaefer300_homotopic_lookup = (schaefer300_homotopic_lookup
                        .iloc[::2, :]
                        .reset_index(drop=True)
                        .rename(columns={0: 'Region'})
                        .assign(Region_Index = lambda x: range(x.shape[0]))
    )

    # Define base calculator
    basecalc = Calculator(configfile=f'{github_base_path}/info_theory_visuals/data/infotheory_measures.yaml')

    # Apply univariate_measures_for_parcellation_file to all parcellation files, concatenating the results into a single DataFrame
    if not os.path.isfile(f'{github_base_path}/info_theory_visuals/data/HCP_7T_four_runs_Schaefer300_homotopic_all_11_infotheory.csv'):

        print(f"Calculating info-theory measures for all parcellation files")

        all_infotheory_measure_results_HCP_7T_fMRI_list = [info_theory_measures_for_parcellation_file(input_parc_file=parc_file,
                                                                                                      basecalc=basecalc,
                                                                                                      schaefer300_homotopic_lookup=schaefer300_homotopic_lookup) 
                                                                                                      for parc_file in parc_files]
        
        # Concatenate and save all the results
        all_infotheory_measure_results_HCP_7T_fMRI = pd.concat(all_infotheory_measure_results_HCP_7T_fMRI_list).reset_index(drop=True)
        all_infotheory_measure_results_HCP_7T_fMRI.to_csv(f'{github_base_path}/info_theory_visuals/data/HCP_7T_four_runs_Schaefer300_homotopic_all_11_infotheory.csv', index=False)
    # else:
    #     print("Loading precomputed univariate measures for all parcellation files...")
    #     all_HCP_7T_fMRI_univariate_results = pd.read_csv(f'{github_base_path}/info_theory_visuals/data/HCP_7T_four_runs_Schaefer300_homotopic_entropy_AIS.csv')
