import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import DatasetMetadata, MetadataManager
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import LeaveOneOut
from sklearn_lvq import GmlvqModel
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.distance_based import ProximityForest
from sktime.classification.kernel_based import RocketClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
import pickle
from tqdm import tqdm
import xgboost as xgb
import os
import warnings
import features

# SKLEARN PLEASE STOP
warnings.filterwarnings("ignore", category=FutureWarning)

# Fixes variability between runs
SEED = 42

# Saves the models for later communication with other files
AVAILABLE_MODELS = {
    # FEATURE-BASED MODELS
    "rf":RandomForestClassifier(
    n_estimators=300,
    random_state=SEED,
    max_features='log2',
    criterion='gini'), # Overall good results with RCE.

    "rf_mini":RandomForestClassifier(
    n_estimators=50,
    random_state=SEED,
    max_features='log2',
    criterion='gini'), # When speed is absolutely necessary. Could be used for RCE.

    "xgb":xgb.XGBClassifier(
        n_estimators=12, #12
        random_state=SEED,
        #max_depth=48, #48 -> 82%
        reg_lambda=1.0, # 1.0
        reg_alpha=1.5, #1.5
        objective="binary:logistic",
    ), # Excellent results with TSFresh features (and without RCE)

    "lvq":GmlvqModel(
        prototypes_per_class=1,
        regularization=0.0,
        max_iter=2500,
        beta=2,
        random_state=SEED,
    ), # Not working properly right now, too slow and low performance.

# TIME SERIES MODELS
    "hive":HIVECOTEV2(
        n_jobs=-1,
        time_limit_in_minutes=5, # Could be a good case use for HABROK or the ML machine.
        verbose=0,
        random_state=SEED,
    ), # Note that there are a ton of small warnings everywhere, may need to fix manually eventually.

    "rocket": RocketClassifier(
        rocket_transform="minirocket",
        n_jobs=-1,
        n_features_per_kernel=8,
        num_kernels=800,
        random_state=SEED,
    ),

    "prox": ProximityForest(
        n_jobs=-1,
        n_estimators=5,
        random_state=SEED,
    ),

}

# Features to cut down to using RFE or FeatBoost.
N_FEATURES = 10


# Main training function
# --------------------------
# arguments:
# dataframe , Data to train with. Should have at the very least Patient ID, Data X, Diagnosis, and Tasks columns.
# run       , Combination of tasks to run, in a list. Example, [13,14] will run for tasks 13 and 14 combined.
# keypoints , Index of keypoints to extract from data. Should be a list of integers.
# model_name, One of AVAILABLE_MODELS.
# n_patients, Number of patients. Required for automatic array dimensions.
# skip      , Skip feature reduction. If set to FALSE, will use RFE to remove features.
# raw       , Use raw time series. If set to FALSE, will extract features using TSFresh.
# project   , Generate PCA projections prior to feature selection. If set to TRUE, will generate components with 75%
#           , variance explanation.
# --------------------------
# returns:
# A dictionary containing the following keys:
# accuracy , A list of training accuracies.
# f1       , Same, but F1.
# precision, Same, but precision.
# recall   ,
# rfe      , RFE selection list, which points to which feature was used in a given run.
# pca      , PCA component loadings for each fold
# feature_origins, Which saves which keypoint/feature originated which tsfresh feature (for explainability).
def train_and_evaluate_loocv(dataframe, run, keypoints, cut_mode, data_length,
                             model_name = "rf",
                             feature_select = "rfe",
                             scheme = "concatenate",
                             vote_mode = "consensus",
                             n_patients=36, skip=None, raw=None, project=True):

    clf = AVAILABLE_MODELS[model_name]

    # Split the data into training and testing sets
    loo = LeaveOneOut()

    # To manage metadata
    metadata_mgr = MetadataManager(file_path = "processed_data/metadata.csv")
    metadata_mgr.load_metadata()

    # Lists to store results for each fold
    feature_importances = []
    accuracies = []
    f1_scores = []
    recalls = []
    precisions = []
    rfe_reports = []
    pca_components = []
    true_label = []
    true_probability = []
    ids = []


    # Prepare the data for split
    X = []


    # Feature generation
    feature_origins = []
    print(f"Feature generation in progress. Targeting {len(keypoints)} time series.")
    if not raw:
        for task in run:
            for keypoint in keypoints:
                params = DatasetMetadata(
                    keypoint = str(keypoint),
                    cut_mode = cut_mode,
                    data_length = data_length,
                    tasks = [task],
                )

                existing = metadata_mgr.find_matching_dataset(params)

                if existing:
                    inbetween = pickle.load(open(f"processed_data/{existing.id}.pkl", "rb"))
                    X.append(inbetween)
                    feature_origins = feature_origins + ["KP"+str(keypoint)] * np.shape(inbetween)[1]
                else:
                    if type(keypoint) is int:
                        dataframe[task][str(keypoint)] = features.make_time_series(dataframe[task]["Data X"], keypoint)
                        inbetween = features.extract_features_from_time_series(dataframe[task],str(keypoint))
                        with open(f"processed_data/{params.id}.pkl", 'wb') as f:
                            pickle.dump(inbetween, f)
                        metadata_mgr.add_dataset(params)
                        X.append(inbetween)
                        feature_origins = feature_origins + ["KP"+str(keypoint)] * np.shape(inbetween)[1]
                    else:
                        distances = []
                        for i in range(0, n_patients):
                            kp = keypoint.split("-")
                            distances.append(features.distance_thumb_index(dataframe[task].iloc[i],
                                                                           index=int(kp[0]),
                                                                           thumb=int(kp[1]),
                                                                           ),
                                             )

                        dataframe[task]["distances"] = distances
                        inbetween = features.extract_features_from_time_series(dataframe[task], "distances")
                        with open(f"processed_data/{params.id}.pkl", 'wb') as f:
                            pickle.dump(inbetween, f)
                        metadata_mgr.add_dataset(params)
                        X.append(inbetween)
                        feature_origins = feature_origins + ["KP"+str(keypoint)] * np.shape(inbetween)[1]


    else:
        for task in run:
            for keypoint in keypoints:
                inbetween = features.make_time_series(dataframe[task]["Data X"], keypoint)
                X.append(inbetween)
                feature_origins = feature_origins + ["KP"+str(keypoint)] * np.shape(inbetween)[1]


    if model_name in ["hive", "rocket"]:
        X = np.stack(X, axis=1) # Train as multivariate
        os.environ["PYTHONWARNINGS"] = "ignore" # It's impossible to look at the console because of  a deprecation warning
    elif scheme == "ensemble":
        X = np.stack(X, axis=1)  # Train as multivariate
    elif scheme == "ensemble_tasks":
        X = [np.hstack(X[i:i+len(keypoints)]) for i in range(0, len(X), len(keypoints))]
        X = np.stack(X, axis=1)
    else:
        X = np.hstack(X)

    print(f"->Features generated and prepared with shape {np.shape(X)}<-")
    print("-------------------------------------------------------------")

    # Prepare target labels
    y = dataframe[task]["Diagnosis"].to_numpy()
    y = (y == "ET").astype(int)
    y = y.ravel()

    # Progress bar
    pbar = tqdm(enumerate(loo.split(X)), desc="LOOCV", total=len(X), ncols=100)

    # Perform LOO-CV
    for j,[train_index, test_index] in pbar:

        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        # Voting scheme
        if scheme == "concatenate":
            # Perform feature selection in split:
            X_train_selected, X_test_selected, loadings, pca_components, rfe_reports, reports = prepare_data(
                X_train, X_test, y_train, rfe_reports, pca_components,
                model_name=model_name,
                skip=skip,
                project=project,
                feature_select=feature_select,
            )

            # Train the model
            clf.fit(X_train_selected, y_train)

            # Predict and evaluate
            y_pred = clf.predict(X_test_selected)

            # Store feature importances
            if model_name in ["rf", "xgb"]:
                feature_importances.append(clf.feature_importances_)
            else:
                feature_importances.append(None)

        elif scheme in ["ensemble","ensemble_tasks"]:
            preds = []
            certainty = []
            for i in range(0,np.shape(X_train)[1]):
                X_ftrain = X_train[:,i,:]
                X_ftest = X_test[:,i,:]
                X_train_selected, X_test_selected, loadings, pca_components, rfe_reports, reports = prepare_data(
                    X_ftrain, X_ftest, y_train, rfe_reports, pca_components,
                    model_name=model_name,
                    skip=skip,
                    project=project,
                    feature_select=feature_select,
                )

                clf.fit(X_train_selected, y_train)
                preds.append(clf.predict(X_test_selected))
                certainty.append(clf.predict_proba(X_test_selected))

            if vote_mode == "consensus":
                y_pred = vote_consensus(X_train, preds, certainty)
            elif vote_mode == "confidence":
                y_pred = vote_confidence(certainty)
            elif vote_mode == "expert":
                y_pred = vote_expert(certainty)

        else:
            raise ValueError("This multi-series scheme is not implemented.")

        # Compute classification metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=1, pos_label=1))
        recalls.append(recall_score(y_test, y_pred, zero_division=1, pos_label=1))
        precisions.append(precision_score(y_test, y_pred, zero_division=1, pos_label=1))
        true_label.append(y_test)
        true_probability.append(clf.predict_proba(X_test_selected))
        ids.append(str(dataframe[task]["Patient ID"].iloc[test_index].values[0]))

        # Update progress bar
        if feature_select == "featboost" and skip == 0:
            pbar.set_postfix(ACC=f"{np.mean(accuracies):.2f}",
                             FT=f"{len(reports)}",
                             PRC=f"{np.mean(precisions):.2f}",
                             REC=f"{np.mean(recalls):.2f}",
                             )

        else:
            pbar.set_postfix(ACC = f"{np.mean(accuracies):.2f}",
                             F1 = f"{np.mean(f1_scores):.2f}",
                             PRC = f"{np.mean(precisions):.2f}",
                             REC = f"{np.mean(recalls):.2f}",
                             )




    # Convert feature importances to a numpy array and calculate the average importances
    #feature_importances = np.array(feature_importances)
    #average_importances = feature_importances.mean(axis=0)

    # Calculate average metrics
    average_accuracy = np.mean(accuracies)
    average_f1 = np.mean(f1_scores)
    average_recall = np.mean(recalls)
    average_precision = np.mean(precisions)

    # Print results
    #print("Average Feature Importances:", average_importances)
    print(f"Average Accuracy: {average_accuracy:.4f}")
    print(f"Average F1 Score: {average_f1:.4f}")
    print(f"Average Recall: {average_recall:.4f}")
    print(f"Average Precision: {average_precision:.4f}")

    return {"accuracy":accuracies,
            "f1":f1_scores,
            "recall":recalls,
            "precision":precisions,
            "rfe":rfe_reports,
            "pca":pca_components,
            "feature_origins":feature_origins,
            "true_labels":true_label,
            "true_probability":true_probability,
            "feature_importances":feature_importances,
            "id":ids,
            }

def prepare_data(
        X_train, X_test, y_train, rfe_reports, pca_components,
        model_name="xgb",
        skip=False,
        project=True,
        feature_select="featboost",
):
    # Perform feature selection in split:
    loadings = None
    reports = None
    if not skip:
        if project:
            X_train, scaler, pcam = features.PCA_transform(X_train)
            X_test = scaler.transform(X_test)
            X_test = pcam.transform(X_test)

        if np.shape(X_train)[1] > N_FEATURES:
            if feature_select == "rfe":
                X_train_selected, X_test_selected, reports = features.rfe_select(
                    X_train, X_test, y_train, AVAILABLE_MODELS["rf"], n_features=N_FEATURES,
                )

            elif feature_select == "featboost":
                selectorXGB = xgb.XGBClassifier(
                    max_depth=12,
                    learning_rate=1.0,
                    n_estimators=24,
                    alpha=2,
                    random_state=SEED,
                )

                X_train_selected, X_test_selected, reports = features.featboost_select(
                    X_train, X_test, y_train, selectorXGB, n_features=N_FEATURES
                )
            else:
                raise ValueError("Feature selection method not implemented.")

            rfe_reports.append(np.array(reports))

            if project:
                loadings = pcam.components_
                component_loadings = loadings[np.where(reports)[0]]
                pca_components.append(component_loadings)
            else:
                pca_components.append(None)

        else:
            X_train_selected = X_train
            X_test_selected = X_test
            loadings = pcam.components_
            pca_components.append(loadings)
            rfe_reports.append(None)
    else:
        if model_name in ["xgb"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        X_train_selected = X_train
        X_test_selected = X_test
        loadings = None
        reports = None
        pca_components.append(None)
        rfe_reports.append(None)

    return X_train_selected, X_test_selected, loadings, pca_components, rfe_reports, reports

def vote_consensus(X_train, preds, certainty):
    proportion = np.round(np.sum(preds) / (np.shape(X_train)[1] / 2))
    if proportion > 1:  # Most have voted ET
        y_pred = [1]
    elif proportion < 1:  # Most have voted CM
        y_pred = [0]
    else:
        vote_confidence_cm = np.sum(np.squeeze(np.array(certainty), axis=1)[:, 0])
        vote_confidence_et = np.sum(np.squeeze(np.array(certainty), axis=1)[:, 1])
        if vote_confidence_et > vote_confidence_cm:
            y_pred = [1]
        else:
            y_pred = [0]

    return y_pred

def vote_confidence(certainty):
    certainty_array = np.array(certainty)
    total_confidence = np.sum(certainty_array, axis=0)  # Sum confidence for each class
    y_pred = [int(np.argmax(total_confidence))]  # 0 for CM, 1 for ET
    return y_pred

def vote_expert(certainty):
    certainty_array = np.array(certainty)  # shape: (n_classifiers, n_classes)
    max_conf_idx = np.unravel_index(np.argmax(certainty_array), certainty_array.shape)
    y_pred = [max_conf_idx[1]]  # index 0 = CM, index 1 = ET
    return y_pred


