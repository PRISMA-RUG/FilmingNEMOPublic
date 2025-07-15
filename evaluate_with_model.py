import classifiers
import features
import utils
import pandas as pd
import json
import os
import warnings

# Stops Future Warnings from SKLearn. Not 100% required but makes Sktime's output basically unreadable when multithreading
# if it isn't.
warnings.filterwarnings("ignore", category=FutureWarning)

# TODO: Make it use command line parameters
# TODO: Automatic number of patient detection

# Usage: This code will run the chosen model with all the defined keypoints for each task separately,
# checking performance and reporting on the best features.

## Define behavior below ## ## ## ##

# Model to evaluate. Hyperparameters pre-adjusted, modify in classifiers.py
MODEL = "xgb"
# Options :
# "rf" - Random Forest. Record: 72% F1.
# "xgb" - XGBoost. Record: 82% F1.
# "lvq" - GMLVQ. Not fully tested yet. Current issues: Slow, errors out by default so I had to hack it to work,
#                unsure if my hacks are correct yet.
# "hive" - HIVECOTEV2. Not fully tested yet. Careful, it's slow.

# Planned:
# CNN/RESNET/Deep Learning using SKTime

FEATURE_SELECTOR = "featboost"
# One of "rfe" or "featboost"

MODE = "features"
# One of:
# features - Calculates all subsets of tasks given
# strict - Calculates only the specific combination of tasks given.

COMBINE = "ensemble_tasks"
VOTING = "confidence"

SKIP = 0 # Skips feature elimination (No RCE, no PCA)
PROJECT = 0 # Performs PCA if = 1, if feature elimination is not skipped. To skip RCE and not PCA, set number of features
# to a really high number (>100).
RAW = 0 # Skips tsfresh feature generation (uses only raw time series). Necessary for some models, like HIVECOTE.

# Tasks to load.
TASKS = ["12", "13", "17","19"]

# Keypoints to use during feature creation. It may be model specific, so check mediapipe_models.py
PAIRS = ["15-16", "15-19", "15-20", "15-21", "15-22",
         "16-19", "16-20", "16-21", "16-22",
         "19-20", "19-21", "19-22",
         "20-21", "20-22",
         "21-22",
         ]
KEYPOINTS = [15,16,19,20,21,22]
KEYPOINTS = KEYPOINTS + PAIRS
#KEYPOINTS = [19,21]

# Data equalization method
DATA_LENGTH = 902 # This is used to pad (or cut!) the raw data.
DATA_CUT_MODE = "copy_butterworth"

## ## ## ## ## ## ## ##

# Pre-generate pandas data frame
if os.path.exists(f"landmarks_{DATA_LENGTH}_{DATA_CUT_MODE}"):
    site_to_load = f"landmarks_{DATA_LENGTH}_{DATA_CUT_MODE}/"
    recut = 0
else:
    site_to_load = "landmarks/"
    recut = DATA_LENGTH

results = utils.load_data(site_to_load,recut, DATA_CUT_MODE)

# Tidies the data so each patient has the data associated with different tasks, plus divides into different dataframes
# for training.
dataframes = utils.generate_tasks(results, TASKS)

# Removes any patient that isn't present in all tasks.
dataframes = utils.equalize_tasks(dataframes)

# Sets task division according to mode
if MODE == "features":
    runs = utils.define_runs(TASKS)
elif MODE == "strict":
    runs = [TASKS]

run_results = []

for run in runs:
    print("RUN FOR TASK "+str(run))
    print(" =========================================================== ")
    print(f"RUN DETAILS: MODEL *{MODEL}*")
    if not SKIP:
        print(f"-- ^ USING SELECTOR {FEATURE_SELECTOR} ^.")
    print(f"|| Projection ({PROJECT}) <|- Skip Selection ({SKIP}) <|- Time Series ({RAW}) ||")
    print(f"|| Task Combination Method: >{COMBINE}< ||")
    if COMBINE != "concatenate":
        print(f"|| || With Voting method  ^{VOTING}^ || ||")
    print(f" ---------------------------------------------------------- ")
    run_results.append(classifiers.train_and_evaluate_loocv(
        dataframes, # Data to use
        run, # Which tasks go for this specific training run
        KEYPOINTS, # Keypoints to extract data from
        DATA_CUT_MODE, # For loading pre-processed features, what type of cut was made
        DATA_LENGTH, # For loading pre-processed features, what length of data was processed
        model_name=MODEL, # Model to use.
        feature_select=FEATURE_SELECTOR,
        n_patients=38, # Number of patients to calculate for.
        skip=SKIP, # Mode selections
        raw=RAW,
        project=PROJECT,
        scheme=COMBINE,
        vote_mode=VOTING,
    ))
    utils.save_run_results(run_results[-1], MODEL, "-".join(run))
    print(" =========================================================== ")




