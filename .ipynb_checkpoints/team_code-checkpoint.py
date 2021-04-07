#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.abspath("./datasets"))
from physionet2021 import PhysioNet2021Dataset
import torch
import logging
# sys.path.append(os.path.abspath("../"))
from resnet import ResNet, BasicBlock, Bottleneck
from torch import nn
from time import time

twelve_lead_model_filename = '12_lead_model.sav'
six_lead_model_filename = '6_lead_model.sav'
three_lead_model_filename = '3_lead_model.sav'
two_lead_model_filename = '2_lead_model.sav'

################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')

    logger = logging.getLogger()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)

    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    data = np.zeros((num_recordings, 14), dtype=np.float32) # 14 features: one feature for each lead, one feature for age, and one feature for sex
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes

    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    model = ResNet(
    block =BasicBlock,
    layers = [2,3,5,7],
    in_channel = 12,
    out_channel = num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss(size_average = True)
    
    leads = twelve_leads
    filename = os.path.join(model_directory, twelve_lead_model_filename)

    params = {'batch_size': 64}

    training_set = PhysioNet2021Dataset(
        data_directory, max_seq_len=4096, ensure_equal_len=True, 
        proc=0,leads = twelve_leads
    )
    train_dataloader = torch.utils.data.DataLoader(training_set, **params)
    model = dataloader_train(train_dataloader, model, device=device, optimizer=optimizer, loss_fn=loss_fn)
    save_model(logger, filename, model, classes)
    
    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    model = ResNet(
    block =BasicBlock,
    layers = [2,3,5,7],
    in_channel = 6,
    out_channel = num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss(size_average = True)
    
    leads = six_leads
    filename = os.path.join(model_directory, six_lead_model_filename)

    params = {'batch_size': 64}

    training_set = PhysioNet2021Dataset(
        data_directory, max_seq_len=4096, ensure_equal_len=True, 
        proc=0,leads = leads
    )
    train_dataloader = torch.utils.data.DataLoader(training_set, **params)
    model = dataloader_train(train_dataloader, model, device=device, optimizer=optimizer, loss_fn=loss_fn)
    save_model(logger, filename, model, classes)

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    model = ResNet(
    block =BasicBlock,
    layers = [2,3,5,7],
    in_channel = 3,
    out_channel = num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss(size_average = True)
    
    leads = three_leads
    filename = os.path.join(model_directory, three_lead_model_filename)

    params = {'batch_size': 64}

    training_set = PhysioNet2021Dataset(
        data_directory, max_seq_len=4096, ensure_equal_len=True, 
        proc=0,leads = leads
    )
    train_dataloader = torch.utils.data.DataLoader(training_set, **params)
    model = dataloader_train(train_dataloader, model, device=device, optimizer=optimizer, loss_fn=loss_fn)
    save_model(logger, filename, model, classes)
    
    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

    
    model = ResNet(
    block =BasicBlock,
    layers = [2,3,5,7],
    in_channel = 2,
    out_channel = num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss(size_average = True)
    
    leads = two_leads
    filename = os.path.join(model_directory, two_lead_model_filename)

    params = {'batch_size': 64}

    training_set = PhysioNet2021Dataset(
        data_directory, max_seq_len=4096, ensure_equal_len=True, 
        proc=0,leads = leads
    )
    train_dataloader = torch.utils.data.DataLoader(training_set, **params)
    model = dataloader_train(train_dataloader, model, device=device, optimizer=optimizer, loss_fn=loss_fn)
    save_model(logger, filename, model, classes)

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
# def save_model(filename, classes, leads, imputer, classifier):
#     # Construct a data structure for the model and save it.
#     d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
#     joblib.dump(d, filename, protocol=0)
    
def save_model(logger, output_directory, model, classes):
    logger.info("Saving model...")

#     cur_sec = int(time())
#     filename = os.path.join(output_directory, f"finalized_model_{cur_sec}.sav")
    data = {'classes': classes, 'model': model}
    joblib.dump(data, output_directory, protocol=0)

    logger.info(f"Saved to {output_directory}")

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    data = joblib.load(filename)
#     model.eval()
    return data

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
#     filename = os.path.join(model_directory, six_lead_model_filename)
    filename = os.path.join(model_directory, six_lead_model_filename)
    data = joblib.load(filename)
    return data

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    data = joblib.load(filename)

    return data

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    data = joblib.load(filename)

    return data

# Generic function for loading a model.
def load_model(filename):
    return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(data, header, recording):
    model = data['model']
    classes = data['classes']
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return run_model(header, recording, model, device, classes, twelve_leads, num_leads=12)

# run_model(header, recording, model, device, classes)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
# def run_model(model, header, recording):
#     classes = model['classes']
#     leads = model['leads']
#     imputer = model['imputer']
#     classifier = model['classifier']

#     # Load features.
#     num_leads = len(leads)
#     data = np.zeros(num_leads+2, dtype=np.float32)
#     age, sex, rms = get_features(header, recording, leads)
#     data[0:num_leads] = rms
#     data[num_leads] = age
#     data[num_leads+1] = sex

#     # Impute missing data.
#     features = data.reshape(1, -1)
#     features = imputer.transform(features)

#     # Predict labels and probabilities.
#     labels = classifier.predict(features)
#     labels = np.asarray(labels, dtype=np.int)[0]

#     probabilities = classifier.predict_proba(features)
#     probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

#     return classes, labels, probabilities

# def run_model(model, header, recording):
    

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
#     rms = np.zeros(num_leads, dtype=np.float32)
#     for i in range(num_leads):
#         x = recording[i, :]
#         rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

    return age, sex, recording

'''
train_dataloader
'''

def dataloader_train(train_dataloader, model, device, optimizer, loss_fn):
    size = len(train_dataloader.dataset)
    for batch, features in enumerate(train_dataloader):
        X = features['sig']
        age = features['age']
        sex = features['sex']
        labels = features['labels']

        age_numpy = age.numpy()
        m = np.nanmean(age_numpy)
        age_numpy = np.nan_to_num(age_numpy, nan=m)
        age = torch.from_numpy(age_numpy)
        age -= age.min()
        age /= age.max()
        demo_features = torch.stack([age, sex.double()], dim= 1)
        demo_features = demo_features.float()
        X, demo_features, labels =  X.to(device), demo_features.to(device), labels.to(device)

        pred = model(X, demo_features)
        loss = loss_fn(pred, labels.float())


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return model

def run_model(header, recording, model, device, classes, leads, num_leads=12):
    size = len(train_dataloader.dataset)
    for batch, features in enumerate(train_dataloader):
        X = features['sig']
        age = features['age']
        sex = features['sex']
        labels = features['labels']
        record_names = feature['record_name']
        
        if age == np.nan:
            age = 20
        
        demo_features = torch.tensor(age, sex.double())
        
        X, demo_features =  X.to(device), demo_features.to(device)
        pred = model(X, demo_features)
        labels = labels.int().numpy()
        pred = pred.numpy()

        labels_list.append(labels)
        pred_list.append(pred)
        
    labels = np.concatenate(labels_list)
    probabilities = np.concatenate(pred_list)
    
    return classes, labels, probabilities
