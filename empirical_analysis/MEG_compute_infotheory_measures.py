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

# MEG preprocessing libraries
import mne
import numpy as np
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import mne_connectivity
from mne_connectivity import envelope_correlation

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
data_dir = "/Users/abry4213/github/info_theory_visuals/data"

# Define configuration file
infotheory_config_file = f"{data_dir}/infotheory_measures.yaml"

# Load SPI groupings
infotheory_measure_info = pd.read_csv(f"{data_dir}/infotheory_measure_info.csv")

# Initialise a Calculator object with this configuration file
basecalc = Calculator(configfile=infotheory_config_file)

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

# Read in brain regions
brain_region_lookup = pd.read_csv(f"{data_dir}/Brain_Region_info.csv", index_col=False).reset_index(drop=True)
base_regions = list(set(brain_region_lookup.Base_Region.tolist()))
source_base_region = "lateraloccipital"

# Check if 
if not op.isfile(f"{data_dir}/MEG_resting_time_series_df.csv"):
    print("MEG_resting_time_series_df.csv not found. Generating it now.")

    # Set downsampling factor
    resample_freq = 80

    data_path = mne.datasets.brainstorm.bst_resting.data_path()
    subjects_dir = op.join(data_path, "subjects")
    subject = "bst_resting"
    trans = op.join(data_path, "MEG", "bst_resting", "bst_resting-trans.fif")
    src = op.join(subjects_dir, subject, "bem", subject + "-oct-6-src.fif")
    bem = op.join(subjects_dir, subject, "bem", subject + "-5120-bem-sol.fif")
    raw_fname = op.join(
        data_path, "MEG", "bst_resting", "subj002_spontaneous_20111102_01_AUX.ds"
    )

    raw = mne.io.read_raw_ctf(raw_fname, verbose="error")
    raw.crop(0, 60).pick_types(meg=True, eeg=False).load_data().resample(resample_freq)
    raw.apply_gradient_compensation(3)
    projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2)
    projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name="MLT31-4407")
    raw.add_proj(projs_ecg + projs_eog)
    raw.apply_proj()
    raw.filter(0.1, None)  # this helps with symmetric orthogonalization later

    cov = mne.compute_raw_covariance(raw)  # compute before band-pass of interest

    ## Compute forward and inverse
    src = mne.read_source_spaces(src)
    fwd = mne.make_forward_solution(raw.info, trans, src, bem, verbose=True)
    del src
    inv = make_inverse_operator(raw.info, fwd, cov)
    del fwd

    # Now we create epochs and prepare to band-pass filter them
    duration = 60.0
    events = mne.make_fixed_length_events(raw, duration=duration)
    tmax = duration - 1.0 / raw.info["sfreq"]
    epochs = mne.Epochs(
        raw, events=events, tmin=0, tmax=tmax, baseline=None, reject=dict(mag=20e-13)
    )
    sfreq = epochs.info["sfreq"]
    del raw, projs_ecg, projs_eog

    labels = mne.read_labels_from_annot(subject, "aparc", subjects_dir=subjects_dir)
    stcs = apply_inverse_epochs(
        epochs, inv, lambda2=1.0 / 9.0, pick_ori="normal", return_generator=True
    )
    label_ts = mne.extract_label_time_course(
        stcs, labels, inv["src"], return_generator=False
    )
    del stcs

    label_ts = np.array(label_ts)  # convert list to array

    n_epochs, n_labels, n_times = label_ts.shape
    label_names = [label.name for label in labels]

    # Flatten into a long DataFrame
    MEG_resting_time_series_df = pd.DataFrame(
        label_ts.reshape(n_epochs * n_labels, n_times),
        columns=[f"time_{i}" for i in range(n_times)],
    )
    MEG_resting_time_series_df["epoch"] = np.repeat(np.arange(n_epochs), n_labels)
    MEG_resting_time_series_df["label"] = np.tile(label_names, n_epochs)

    MEG_resting_time_series_df = (MEG_resting_time_series_df
        .melt(id_vars=["epoch", "label"], var_name="time", value_name="value")
        .assign(time = lambda x: x["time"].str.replace("time_", "").astype(int)))
    
    # Save the DataFrame to a CSV file
    MEG_resting_time_series_df.to_csv(f"{data_dir}/MEG_resting_time_series_df.csv", index=False)

else:
    print("MEG_resting_time_series_df.csv found. Loading it now.")
    MEG_resting_time_series_df = pd.read_csv(f"{data_dir}/MEG_resting_time_series_df.csv")
    MEG_resting_time_series_df["time"] = MEG_resting_time_series_df["time"].astype(int)

# Take the substring before the first hyphen
MEG_resting_time_series_df = (MEG_resting_time_series_df.assign(Base_Region = lambda x: x["label"].str.split("-").str[0],
                                  Hemisphere = lambda x: x["label"].str.split("-").str[1])
                            .assign(Hemisphere = lambda x: np.where(x["Hemisphere"] == "lh", "Left", "Right"))
                                  )

# Subset to left hemisphere
MEG_resting_time_series_df_subset = (MEG_resting_time_series_df
    .query("Hemisphere == 'Left'"))

###################################################################################################
def compute_info_theory_SPIs_for_dataset(dataset_name, output_data_dir, input_dataframe, basecalc, brain_region_lookup, source_base_region):
    if os.path.isfile(f"{output_data_dir}/{dataset_name}_infotheory_measures.csv"):
        return 

    # Extract the 34 base cortical regions
    base_regions = list(set(brain_region_lookup.Base_Region.tolist()))

    # Create a list to store results
    this_dataset_infotheory_results_list = []

    # Extract the time series for the base region
    source_TS = input_dataframe.query("Base_Region==@source_base_region")['value'].values

    # z-score the time series
    source_TS = zscore(source_TS)

    # Convert to a Java array
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
                                                .assign(Base_Region = source_base_region))
    this_dataset_infotheory_results_list.append(univariate_dataframe_res)

    # Iterate over the base regions as the target regions
    for target_base_region in base_regions:
        # Skip the source region
        if target_base_region == source_base_region:
            continue

        # Set the source
        random.seed(127)

        # Subset the input time series data to this region
        target_TS = input_dataframe.query("Base_Region==@target_base_region")['value'].values

        # z-score the time series
        target_TS = zscore(target_TS)

        # Convert to a Java array
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
                                                    .assign(Base_Region = target_base_region))
        this_dataset_infotheory_results_list.append(univariate_dataframe_res)

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
                        .assign(Measure_Type = "Bivariate",
                                Measure = SPIs)
        )
        
        # Append this region's info theory SPI data to the list
        this_dataset_infotheory_results_list.append(this_SPI_subject_calc_res)

    # Concatenate the list of dataframes into a single dataframe
    this_dataset_infotheory_results = pd.concat(this_dataset_infotheory_results_list)

    # Save to a CSV file
    this_dataset_infotheory_results.to_csv(f"{output_data_dir}/{dataset_name}_infotheory_measures.csv", index=False)


# Compute the info theory SPIs for this dataset
compute_info_theory_SPIs_for_dataset(dataset_name=f"MEG_Resting_State_Epoch_1min", output_data_dir=data_dir, 
                                    input_dataframe=MEG_resting_time_series_df_subset, basecalc=basecalc, 
                                    brain_region_lookup=brain_region_lookup, source_base_region=source_base_region)

# Shut down the JVM at the end of session
jpype.shutdownJVM() 