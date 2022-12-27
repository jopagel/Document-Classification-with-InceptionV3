import pathlib
import random
import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image, ImageReadMode
import copy
import pathlib
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler



def get_file_paths_and_labels(data_root):
    """
    Returns a dataframe with the columns "path" and "label" corresponding to each image in the data root.

    Parameters:
    -------
    data_root: str
    path to the dataset

    Returns
    -------
    labels_df: pd.DataFrame
    dataframe with the columns "path" and "label" corresponding to each image in the data root
    label_to_index: dict
    dictionary that maps the numerical class label back to the document name
    """
    image_paths = sorted([str(path).split("jpg/")[1] for path in data_root.glob('*/*.jpg')])
    random.shuffle(image_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    labels = [label_to_index[pathlib.Path(path).parent.name] for path in image_paths]
    labels_df = pd.DataFrame({"path": image_paths, "label": labels })

    return labels_df, label_to_index


def initialize_model(num_classes=10, use_pretrained=True):
    model_ft = models.inception_v3(pretrained=use_pretrained)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)
    input_size = 299

    return model_ft, input_size