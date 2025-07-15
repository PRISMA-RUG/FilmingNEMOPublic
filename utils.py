import pandas as pd
import itertools
import json
from datetime import datetime
import numpy as np
import os
import csv
import ast
from pathlib import Path
import filters
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict, field
import hashlib

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return super().default(obj)

def generate_tasks(df, tasks):
    df_dict = {}
    for task in tasks:
        df_dict[task] = df[df["Task"]==int(task)].copy()

    return df_dict

def equalize_tasks(df_dict):
    tasks = df_dict.keys()
    for task_to_compare in tasks:
        for task_to_keep in tasks:
            vals_to_keep = df_dict[task_to_keep]["Patient ID"].to_list()
            df_dict[task_to_compare] = df_dict[task_to_compare][df_dict[task_to_compare]["Patient ID"].isin(vals_to_keep)]

    return df_dict

def define_runs(tasks):
    subsets = []
    for r in range(1, len(tasks) + 1):
        subsets.extend(itertools.combinations(tasks, r))
    return [list(subset) for subset in subsets]

def save_run_results(run_results, model, tasks):
    # Ensure that the Results directory exists
    os.makedirs("Results", exist_ok=True)

    with open(f'Results/{datetime.today().strftime("%d%m%y")}_{model}_{tasks}.json', 'w') as f:
        json.dump(run_results, f, cls=NumpyEncoder)

    return 1

def load_data(location, recut=0, mode="copy"):
    results = pd.DataFrame(columns=["Patient ID", "Task", "Diagnosis",
                                    "Data X", "Data Y", "Data Z", "Duration"])

    # Location of landmarks to load
    files = os.listdir(location)

    # Load landmarks and put in dataframe.
    # Special check: If .json, it's raw keypoints. If .pkl then it's a pre-loaded and processed so we don't need a loop.
    for file in files:
        if file.endswith(".json"):
            ldmrks = json.load(open(location + file))
            results.loc[len(results)] = [ldmrks["patient_id"], ldmrks["task_id"], ldmrks["diagnosis"], ldmrks["x"],
                                         ldmrks["y"], ldmrks["z"], len(ldmrks["x"])]
        elif file.endswith(".pkl"):
            results = pd.read_pickle(location + file)

    if recut:
        print("Initiating time series reprocessing! Duration will be modified.")
        print(f"(Saving processed results to landmarks_{recut}_{mode}/. Use this as location for next run)")

        if mode == "copy":
            results = ExtendDataCopy("Data X", results, recut).extend(filter=filters.nf)
            results = ExtendDataCopy("Data Y", results, recut).extend(filter=filters.nf)
            results = ExtendDataCopy("Data Z", results, recut).extend(filter=filters.nf)

        elif mode == "pad":
            results = ExtendDataPad("Data X", results, recut).extend(filter=filters.nf)
            results = ExtendDataPad("Data Y", results, recut).extend(filter=filters.nf)
            results = ExtendDataPad("Data Z", results, recut).extend(filter=filters.nf)

        elif mode == "copy_butterworth":
            results = ExtendDataCopy("Data X", results, recut).extend(filter=filters.butter_lowpass_filter)
            results = ExtendDataCopy("Data Y", results, recut).extend(filter=filters.butter_lowpass_filter)
            results = ExtendDataCopy("Data Z", results, recut).extend(filter=filters.butter_lowpass_filter)

        elif mode == "copy_detrend_butterworth":
            results = ExtendDataCopy("Data X", results, recut).extend(filter=filters.butter_detrend)
            results = ExtendDataCopy("Data Y", results, recut).extend(filter=filters.butter_detrend)
            results = ExtendDataCopy("Data Z", results, recut).extend(filter=filters.butter_detrend)


        else:
            raise ValueError("Invalid mode.")


        # Create the directory if it does not exist
        output_dir = f"landmarks_{recut}_{mode}/"
        os.makedirs(output_dir, exist_ok=True)

        # Save the DataFrame to a CSV file
        results.to_pickle(os.path.join(output_dir, f"processed_landmarks_{recut}_{mode}.pkl"))

    return results


@dataclass
class DatasetMetadata:
    keypoint: str
    cut_mode: str
    data_length: int
    tasks: List[str]

    @property
    def id(self) -> str:
        """Generate a unique hash ID based on all parameters"""
        hash_input = json.dumps({
            'keypoint': self.keypoint,
            'cut_mode': self.cut_mode,
            'data_length': self.data_length,
            'tasks': sorted(self.tasks),
        }, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]  # Use first 16 chars of hash

    @classmethod
    def from_dict(cls, data: Dict):
        """Create instance from dictionary with string values"""
        return cls(
            keypoint=data['keypoint'],
            cut_mode=data['cut_mode'],
            data_length=int(data['data_length']),
            tasks=ast.literal_eval(data['tasks']),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary with serialized values for CSV"""
        return {
            'id': self.id,
            'keypoint': str(self.keypoint),
            'cut_mode': self.cut_mode,
            'data_length': str(self.data_length),
            'tasks': str(self.tasks),
        }


class MetadataManager:
    def __init__(self, file_path: str = 'metadata.csv'):
        self.file_path = Path(file_path)
        self.metadata_records: Dict[str, DatasetMetadata] = {}  # key is the auto-generated ID

    def load_metadata(self):
        """Load metadata from CSV file if it exists"""
        if not self.file_path.exists():
            return

        with open(self.file_path, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata = DatasetMetadata.from_dict(row)
                self.metadata_records[metadata.id] = metadata

    def add_dataset(self, metadata: DatasetMetadata) -> str:
        """
        Add a dataset to the metadata records and save to CSV
        Returns the generated ID for the dataset
        """
        dataset_id = metadata.id
        self.metadata_records[dataset_id] = metadata
        self.save_metadata()
        return dataset_id

    def save_metadata(self):
        """Save current metadata to CSV file"""
        if not self.metadata_records:
            return

        fieldnames = ['id', 'keypoint', 'cut_mode', 'data_length', 'tasks']

        with open(self.file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metadata in self.metadata_records.values():
                writer.writerow(metadata.to_dict())

    def find_matching_dataset(self, metadata: DatasetMetadata) -> Optional[DatasetMetadata]:
        """
        Check if we already have a dataset with these exact parameters
        Returns the existing metadata if found, None otherwise
        """
        return self.metadata_records.get(metadata.id)

    def get_dataset_by_id(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Retrieve metadata by its ID"""
        return self.metadata_records.get(dataset_id)

class ExtendDataCopy:
    def __init__(self, series, results, recut):
        self.series = series
        self.results = results
        self.recut = recut

    def extend(self, filter):
        for i, time_series in enumerate(self.results[self.series]):
            filtered_series = filter(time_series)
            # Repeat along the time dimension
            repeats = (self.recut // np.shape(filtered_series)[0]) + 1  # Ensure enough repetitions
            extended_data = np.tile(filtered_series, (repeats, 1))  # Tile along axis 0

            # Trim excess to match exact size
            self.results.at[i, self.series] = np.array(extended_data[:self.recut, :])

        return self.results

class ExtendDataPad(ExtendDataCopy):
    def extend(self, filter):
        for i, time_series in enumerate(self.results[self.series]):
            time_series = filter(time_series)
            extended_data = np.pad(
                time_series,
                ((0, self.recut), (0, 0)),
                mode="constant",
                constant_values=0,
            )  # Pad along axis 0

            # Trim excess to match exact size
            self.results.at[i, self.series] = np.array(extended_data[:self.recut, :])

        return self.results
