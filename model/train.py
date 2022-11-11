import sys
import logging
import warnings
from collections import Counter

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from model import Net

warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Hyperparameters
batch_size = 10
learning_rate = 0.01
n_epochs = 30

torch.manual_seed(24)

data_set_folder = "/mnt/disks/disk1/ImgScreenshots"
model_path = "/mnt/disks/disk1"
data_set_folder = "/tmp/ImgScreenshots"
model_path = "/Users/vn53q3k/OneDriveWalmartInc/MLPython/dl-test-automation/trained_models"
# out_model = f'{model_path}/model_ta_{strftime("%Y.%m.%d.%H.%M.%S", gmtime())}.pt'
out_model = f"{model_path}/model_taObjects.pt"


# data_set_folder = "/home/jupyter/data/screenshots2/screenshots"
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()


def fetch_dataset(data_set_folder):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((224, 224))]
    )
    dataset = torchvision.datasets.ImageFolder(root=data_set_folder, transform=transform)
    return dataset


def split_dataset(dataset):
    X = []
    y = []
    for idx in range(len(dataset)):
        input, target = dataset.__getitem__(idx)
        X.append(input)
        y.append(target)
    return train_test_split(X, y, test_size=0.1, stratify=y)


def fetch_dataloaders(X_train, y_train, X_test, y_test):
    train_dataset = TensorDataset(torch.stack(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.stack(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader, test_loader


def main():
    if not train_on_gpu:
        logging.info("CUDA is not available.  Training on CPU ...")
    else:
        logging.info("CUDA is available!  Training on GPU ...")

    dataset = fetch_dataset(data_set_folder)
    logging.info(f"Avaialble classes {dataset.class_to_idx}")
    logging.info(f"Class distribution {dict(Counter(dataset.targets))}")

    X_train, X_test, y_train, y_test = split_dataset(dataset)
    logging.info(f"Total Train features {len(X_train)} and output {len(y_train)}")
    logging.info(f"Total Test features {len(X_test)} and output {len(y_test)}")
    logging.info(f"Class distribution after split in Train dataset {dict(Counter(y_train))}")
    logging.info(f"Class distribution after split in Test dataset {dict(Counter(y_test))}")

    train_loader, test_loader = fetch_dataloaders(X_train, y_train, X_test, y_test)

    model = Net(len(dataset.classes))
    logging.info(model)

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        logging.info("Moving model to GPU ... ")
        model.cuda()

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    logging.info("Starting training ... ")
    for epoch in range(1, n_epochs + 1):
        # keep track of training and validation loss
        train_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # logging.info(target)
            # calculate the batch loss
            loss = criterion(output, target.long())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.sampler)
        # print training/validation statistics
        if epoch % 10 == 0:
            logging.info("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))

    logging.info("Done training ... ")
    torch.save(model.state_dict(), out_model)
    logging.info(f"Model is saved with name {out_model}")

    # track test loss
    test_loss = 0.0
    classes = dataset.classes
    class_correct = list(0.0 for i in range(len(dataset.classes)))
    class_total = list(0.0 for i in range(len(dataset.classes)))

    actual = []
    predictions = []

    model.eval()
    # iterate over test data
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target.long())
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(target.shape[0]):
            label = target.data[i]
            class_correct[label.long()] += correct[i].item()
            class_total[label.long()] += 1
            # for confusion matrix
            actual.append(classes[target.data[i].long().item()])
            predictions.append(classes[pred.data[i].item()])

    # plot confusion matrix
    cm = confusion_matrix(actual, predictions, labels=classes)
    logging.info(classification_report(actual, predictions))
    cmp = ConfusionMatrixDisplay(cm, classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmp.plot(ax=ax, xticks_rotation="vertical")

    # average test loss
    test_loss = test_loss / len(test_loader.dataset)
    logging.info("Test Loss: {:.6f}\n".format(test_loss))

    for i in range(len(dataset.classes)):
        if class_total[i] > 0:
            logging.info(
                "Test Accuracy of %5s: %2d%% (%2d/%2d)"
                % (
                    classes[i],
                    100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]),
                    np.sum(class_total[i]),
                )
            )
        else:
            logging.info("Test Accuracy of %5s: N/A (no training examples)" % (classes[i]))

    logging.info(
        "\nTest Accuracy (Overall): %2d%% (%2d/%2d)"
        % (100.0 * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total))
    )


if __name__ == "__main__":
    main()
