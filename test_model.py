#!/usr/bin/env python

# Do *not* edit this script.

import numpy as np, os, sys
from team_code import load_twelve_lead_model, load_six_lead_model, load_three_lead_model, load_two_lead_model
from team_code import run_twelve_lead_model, run_six_lead_model, run_three_lead_model, run_two_lead_model
from helper_code import *
from physionet2021 import PhysioNet2021Dataset
import torch

def run_leads_model(leads_model, leads):
    params = {'batch_size': 64}
    test_set = PhysioNet2021Dataset(data_directory, leads=leads)
    test_dataloader = torch.utils.data.DataLoader(test_set, **params)
    
    labels_list = []
    pred_list = []
    record_names_list = []
    size = len(test_dataloader.dataset)
    classes = leads_model['classes']
    model = leads_model['model']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for batch, features in enumerate(test_dataloader):
            X = features['sig']
            age = features['age']
            sex = features['sex']
            labels = features['labels']
            record_names = features['record_name']

            age_numpy = age.numpy()
            m = np.nanmean(age_numpy)
            age_numpy = np.nan_to_num(age_numpy, nan=m)
            age = torch.from_numpy(age_numpy)

            age -= age.min()
            age /= age.max()
            demo_features = torch.stack([age, sex.double()], dim= 1)
            demo_features = demo_features.float()

            X, demo_features =  X.to(device), demo_features.to(device)
            pred = model(X, demo_features)
            labels = labels.int().numpy()
            pred = pred.numpy()

            labels_list.append(labels)
            pred_list.append(pred)
            record_names_list.extend(record_names)
#             break
    labels = np.concatenate(labels_list)
    probabilities = np.concatenate(pred_list)

    for record_name_idx in range(len(record_names_list)):
        # Save model outputs.
        head, tail = os.path.split(record_names_list[record_name_idx])
        root, extension = os.path.splitext(tail)
        output_file = os.path.join(output_directory, root + '.csv')
        save_outputs(output_file, classes, labels[record_name_idx], probabilities[record_name_idx])
    print ('Done')

# Test model.
def test_model(model_directory, data_directory, output_directory):
    # Load model.
    print('Loading models...')

    twelve_lead_model = load_twelve_lead_model(model_directory)
    six_lead_model = load_six_lead_model(model_directory)
    three_lead_model = load_three_lead_model(model_directory)
    two_lead_model = load_two_lead_model(model_directory)

    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the outputs if it does not already exist.
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Run model for each recording.
    print('Running model...')

    run_leads_model(twelve_lead_model, twelve_leads)
    run_leads_model(six_lead_model, six_leads)
    run_leads_model(three_lead_model, three_leads)
    run_leads_model(two_lead_model, two_leads)
    

#     params = {'batch_size': 64}
#     test_set = PhysioNet2021Dataset(data_directory)
#     test_dataloader = torch.utils.data.DataLoader(test_set, **params)
    
#     labels_list = []
#     pred_list = []
#     record_names_list = []
#     size = len(test_dataloader.dataset)
#     classes = twelve_lead_model['classes']
#     model = twelve_lead_model['model']
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     with torch.no_grad():
#         for batch, features in enumerate(test_dataloader):
#             X = features['sig']
#             age = features['age']
#             sex = features['sex']
#             labels = features['labels']
#             record_names = features['record_name']

#             age_numpy = age.numpy()
#             m = np.nanmean(age_numpy)
#             age_numpy = np.nan_to_num(age_numpy, nan=m)
#             age = torch.from_numpy(age_numpy)

#             age -= age.min()
#             age /= age.max()
#             demo_features = torch.stack([age, sex.double()], dim= 1)
#             demo_features = demo_features.float()

#             X, demo_features =  X.to(device), demo_features.to(device)
#             pred = model(X, demo_features)
#             labels = labels.int().numpy()
#             pred = pred.numpy()

#             labels_list.append(labels)
#             pred_list.append(pred)
#             record_names_list.extend(record_names)
# #             break
#     labels = np.concatenate(labels_list)
#     probabilities = np.concatenate(pred_list)

#     for record_name_idx in range(len(record_names_list)):
#         # Save model outputs.
#         head, tail = os.path.split(record_names_list[record_name_idx])
#         root, extension = os.path.splitext(tail)
#         output_file = os.path.join(output_directory, root + '.csv')
#         save_outputs(output_file, classes, labels[record_name_idx], probabilities[record_name_idx])

#     print('Done.')

if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 4:
        raise Exception('Include the model, data, and output folders as arguments, e.g., python test_model.py model data output.')

    model_directory = sys.argv[1]
    data_directory = sys.argv[2]
    output_directory = sys.argv[3]

    test_model(model_directory, data_directory, output_directory)
