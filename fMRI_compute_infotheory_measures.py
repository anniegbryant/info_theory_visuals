import pandas as pd
import numpy as np
import random

# Java/pyspi
import jpype
from pyspi.calculator import Calculator

# File handling
from copy import deepcopy
from glob import glob
import os
import os.path as op

# Statistics
from scipy.stats import zscore

###################################################################################################
# Point to pyspi installation of infodynamics
jarLocation = "/Users/abry4213/github/pyspi/pyspi/lib/jidt/infodynamics.jar"

# Check if a JVM has already been started
# If not, start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# Set seed to 127
random.seed(127)


###################################################################################################
# Define data directory

# Set seed to 127
random.seed(127)

# Define data directory

# Define configuration file
infotheory_config_file = "data/infotheory_measures.yaml"

# Load SPI groupings
infotheory_measure_info = pd.read_csv("data/infotheory_measure_info.csv")

# Read in brain regions
brain_region_lookup = pd.read_csv("data/Brain_Region_info.csv", index_col=False).reset_index(drop=True)
base_regions = list(set(brain_region_lookup.Base_Region.tolist()))
source_base_region = "lateraloccipital"

# Initialise a Calculator object with this configuration file
basecalc = Calculator(configfile=infotheory_config_file)

# Create a Gaussian entropy calculator
entropy_calcClass = jpype.JPackage("infodynamics.measures.continuous.kozachenko").EntropyCalculatorMultiVariateKozachenko
entropy_calc = entropy_calcClass()
entropy_calc.initialise()

# Create an Active Information Storage calculator with KSG (Kraskov) density estimator
AIScalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
AIScalc = AIScalcClass()
AIScalc.initialise()

# Add observations
AIScalc.setProperty("NORMALISE", "true")

def compute_info_theory_SPIs_for_subject(subject_ID, basecalc, brain_region_lookup, source_base_region="lateraloccipital"):
    output_file = f"data/HCP_{subject_ID}_rsfMRI_infotheory_measures.csv"
    if os.path.isfile(output_file):
        return 
    
    # Load the subject time series data
    subject_data = pd.read_csv(f"data/HCP_{subject_ID}_rsfMRI_DesikanKilliany_TS.csv").values

    # Extract the 34 base cortical regions
    base_regions = list(set(brain_region_lookup.Base_Region.tolist()))

    # Create a list to store results
    this_subject_infotheory_results_list = []

    # Extract the time series for the base region
    source_index = brain_region_lookup.query("Base_Region == @source_base_region & Hemisphere == 'Left'").Region_Index.tolist()[0]
    source_TS = subject_data[:, source_index]

    # z-score the time series
    source_TS = zscore(source_TS)

    # Convert to Java array
    source_TS_Array = jpype.JArray(jpype.JDouble)([float(x) for x in source_TS])

    # Compute entropy for the source region
    source_entropy_calc = deepcopy(entropy_calc)
    source_entropy_calc.setObservations(source_TS_Array)
    source_entropy = source_entropy_calc.computeAverageLocalOfObservations()

    # Compute AIS for the source region
    source_AIS_calc = deepcopy(AIScalc)
    source_AIS_calc.setObservations(source_TS_Array)
    source_AIS = source_AIS_calc.computeAverageLocalOfObservations()

    # Add to dataframe
    univariate_dataframe_res = (pd.DataFrame({"Measure": ["entropy_kozachenko", "AIS_kraskov"],
                                                "region_from": [source_base_region, source_base_region],
                                                "region_to": [source_base_region, source_base_region],
                                                "Measure_Type": ["Univariate", "Univariate"],
                                                "value": [source_entropy, source_AIS]})
                                                .assign(Base_Region = source_base_region,
                                                        Sample_ID = subject_ID))
    this_subject_infotheory_results_list.append(univariate_dataframe_res)

    # Iterate over the base regions as the target regions
    for target_base_region in base_regions:
        # Skip the source region
        if target_base_region == source_base_region:
            continue

        # Set the source
        random.seed(127)

        # Find the index for the target region, left hemisphere
        target_index = brain_region_lookup.query("Base_Region == @target_base_region & Hemisphere == 'Left'").Region_Index.tolist()[0]
            
        print("Target index: ", target_index)

        # Subset the subject_data numpy array to just target_index
        target_TS = subject_data[:, target_index]

        # z-score the time series
        target_TS = zscore(target_TS)

        # Convert to Java array
        target_TS_Array = jpype.JArray(jpype.JDouble)([float(x) for x in target_TS])

        ################# Univariate #################
        # Compute entropy for the target region
        target_entropy_calc = deepcopy(entropy_calc)
        target_entropy_calc.setObservations(target_TS_Array)
        target_entropy = target_entropy_calc.computeAverageLocalOfObservations()

        # Compute AIS for the target region
        target_AIS_calc = deepcopy(AIScalc)
        target_AIS_calc.setObservations(target_TS_Array)
        target_AIS = target_AIS_calc.computeAverageLocalOfObservations()

        # Compute univariate measures for the target region
        univariate_dataframe_res = (pd.DataFrame({"Measure": ["entropy_kozachenko", "AIS_kraskov"],
                                                    "region_from": [target_base_region, target_base_region],
                                                    "region_to": [target_base_region, target_base_region],
                                                    "Measure_Type": ["Univariate", "Univariate"],
                                                    "value": [target_entropy, target_AIS]})
                                                    .assign(Base_Region = target_base_region,
                                                            Sample_ID = subject_ID))
        this_subject_infotheory_results_list.append(univariate_dataframe_res)

        ################# Bivariate #################
        source_TS = source_TS.reshape(1, -1)
        target_TS = target_TS.reshape(1, -1)

        # Combine left_TS and right_TS into a 2 by 1200 numpy array
        bilateral_arr_to_compute = np.concatenate((source_TS, target_TS), axis=0)

        # Make a copy of calc and compute
        this_SPI_subject_calc = deepcopy(basecalc)
        this_SPI_subject_calc.load_dataset(bilateral_arr_to_compute)
        this_SPI_subject_calc.compute()

        # Flatten the MultiIndex columns to just the process IDs
        this_SPI_subject_calc_res = deepcopy(this_SPI_subject_calc.table)
        SPIs = this_SPI_subject_calc_res.columns.get_level_values(0).unique()
        this_SPI_subject_calc_res.columns = this_SPI_subject_calc_res.columns.get_level_values(1)

        # Pivot and clean up this region's info theory SPI data, adding the SPI column at the end
        this_SPI_subject_calc_res = (this_SPI_subject_calc_res
                        .reset_index(level=0) # Convert index to column
                        .rename(columns={"index": "region_from"}) # Rename index as first hemisphere
                        .melt(id_vars="region_from") # Pivot data from wide to long
                        .rename(columns={"process": "region_to"}) # Rename hemisphere receiving the connection
                        .assign(region_from=lambda x: x['region_from'].replace({'proc-0': source_base_region, 'proc-1': target_base_region}),
                                region_to=lambda x: x['region_to'].replace({'proc-0': source_base_region, 'proc-1': target_base_region}))
                        .query("region_from == @source_base_region & region_to == @target_base_region") 
                        .assign(Sample_ID = subject_ID,
                                Measure_Type = "Bivariate",
                                Measure = SPIs)
        )
        
        # Append this region's info theory SPI data to the list
        this_subject_infotheory_results_list.append(this_SPI_subject_calc_res)

    # Concatenate the list of dataframes into a single dataframe
    this_subject_infotheory_results = pd.concat(this_subject_infotheory_results_list)

    # Save to a CSV file
    this_subject_infotheory_results.to_csv(output_file, index=False)


# Compute info theory measures for a specific subject
subject_ID = "298051"
compute_info_theory_SPIs_for_subject(subject_ID, basecalc, brain_region_lookup, source_base_region=source_base_region)

# Shut down the JVM at the end of session
jpype.shutdownJVM() 