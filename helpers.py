import pandas as pd
import os
import shutil
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.io import read_image, ImageReadMode
import pathlib
import random
import torch
from torch.utils.data import Dataset
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
    if os.name == "nt":
        delimiter = "\\"
    else:
        delimiter = "/"

    image_paths = sorted([str(path).split("jpg" + delimiter)[1] for path in data_root.glob("*" + delimiter + "*.jpg")])
    random.shuffle(image_paths)
    label_names = sorted(item.name for item in data_root.glob('*\\') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    labels = [label_to_index[pathlib.Path(path).parent.name] for path in image_paths]
    labels_df = pd.DataFrame({"path": image_paths, "label": labels})
    return labels_df, label_to_index


def initialize_model(num_classes=10, use_pretrained=True):
    """
    Initialize InceptionV3 with a specified number of output classes.

    Parameters:
    -------
    num_classes: the number of document classes
    use_pretrained: specify whether to use the InceptionV3 with pretrained weights.

    Returns
    -------
    model_ft: the initialized InceptionV3 model
    """
    model_ft = models.inception_v3(pretrained=use_pretrained)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft


class CustomImageDataset(Dataset):
    """
    Custom Pytorch Dataset.
    """

    def __init__(self, transform, labels_df, image_dir):
        self.img_labels = labels_df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.RGB)
        image = self.transform(image)
        label = self.img_labels.iloc[idx, 1]

        return image, label


def train_inception(config, train_dataset, test_dataset):
    """
    Training procedure of InceptionV3.

    Parameters:
    -------
    config: configuration of model training

    """

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], sampler=val_sampler
    )

    dataloaders = {"train": train_loader, "val": val_loader}

    net, input_size = initialize_model()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=config["lr"],
        momentum=config["mom"],
        weight_decay=config["weight_decay"],
    )

    for epoch in range(config["num_epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(dataloaders["train"], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, aux_outputs = net(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 8000 == 7999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(dataloaders["val"], 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, aux_outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            print(f"The path is: {path}")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


def test_accuracy(net, config, test_dataset, test_sampler, device="cuda"):
    """
    Function to test the the trained model on the test dataset

    Parameters:
    -------
    hyperparameter_grid: the hyperparameter grid
    num_samples: the number of models to sample hyperparameters for.
    max_num_epochs: the number of epochs for which the models are to be trained.
    gpus: the number of gpus for model training.

    """

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"][0], sampler=test_sampler
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, aux_outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(hyperparameter_grid, save_dir, num_samples=10, max_num_epochs=15, gpus_per_trial=1):
    """
    Function to tune the hyperparameters with Ray.

    Parameters:
    -------
    hyperparamter_grid: the hyperparameter grid
    num_samples: the number of models to sample hyperparameters for.
    max_num_epochs: the number of epochs for which the models are to be trained.
    gpus: the number of gpus for model training.

    """

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        train_inception,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=hyperparameter_grid,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print(
        "Best trial final validation accuracy: {}F".format(
            best_trial.last_result["accuracy"]
        )
    )
    shutil.copytree(best_checkpoint_dir, save_dir)

