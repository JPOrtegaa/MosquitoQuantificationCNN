import copy
import os
import pdb
import sys
import argparse

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import SubsetRandomSampler

import pandas as pd


def transform_images(train_folder):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    images_transformed = datasets.ImageFolder(os.path.join(train_folder), transform=data_transform)

    return images_transformed

# Setting up the model for training purposes
def setting_model(cnn_model_name):
    if cnn_model_name == 'resnet50':
        if pre_trained == 'PreTrained':
          cnn_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
          cnn_model = models.resnet50(weights=None)
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device) # Device from main (GPU or CPU)

        num_features = cnn_model.module.fc.in_features
        cnn_model.module.fc = torch.nn.Linear(num_features, 2) # 2 because it's binary classification

    elif cnn_model_name == 'vgg16':
        if pre_trained == 'PreTrained':
          cnn_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
          cnn_model = models.vgg16(weights=None)
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)

        num_features = cnn_model.module.classifier[6].in_features
        cnn_model.module.classifier[6] = torch.nn.Linear(num_features, 2)
    else:
        print('Invalid model')
        sys.exit(1)

    cnn_model = cnn_model.cuda()

    return cnn_model

def training_parameters():
    criteria = torch.nn.CrossEntropyLoss()
    optimize = torch.optim.Adam(model.parameters(), lr=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimize, step_size=15, gamma=0.75)

    return criteria, optimize, lr_scheduler

def train_model(model_cnn, criterion_train, optimizer_train, scheduler_train, epochs):

    # Number of folds that the dataset was divided
    num_folds = len(folds)

    # Variables to pick the best weights of the model
    best_acc = 0.0
    best_eval_acc = 0.0

    # Dataframe to save the scores obtained from validation
    val_scores = pd.DataFrame(columns=['score', 'class'])

    # Lists of the probabilities and it's respective class from validation
    pos_probabilities = []
    validation_labels = []

    # Iteration through folds
    for fold in range(num_folds):
        model_cnn.train()
        # Training with one fold for n epochs
        for epoch in range(epochs):
            acc = 0.0
            loss = 0.0

            # Obtaining images and respective labels from the folder at batch's size
            for inputs, labels in dataloaders[fold]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero gradients for training
                optimizer_train.zero_grad()

                outputs = model_cnn(inputs)
                _, predictions = torch.max(outputs, 1)
                loss = criterion_train(outputs, labels)

                # Update model parameters in the direction that minimizes the loss
                loss.backward()
                optimizer_train.step()
                scheduler_train.step()

                # Counting the hits and misses from the predictions
                acc += torch.sum(predictions == labels.data)
                loss += loss.item() * inputs.size(0)

            # Calculating the epoch's accuracy and loss
            acc /= folds[fold]
            loss /= folds[fold]
            print(f'Epoch {epoch} accuracy = {acc}')
            print(f'Epoch {epoch} loss = {loss}')

            # Updating best model
            if acc > best_acc:
                print(f'Best model updated Epoch {epoch}: {acc}')
                best_acc = acc

        # After training, evaluate the model with the other folder of images
        model_cnn.eval()
        if fold == 0:
            eval_fold = 1
        else:
            eval_fold = 0

        eval_acc = 0.0
        with torch.no_grad():
            for inputs, labels in dataloaders[eval_fold]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model_cnn(inputs)
                _, predictions = torch.max(outputs, 1)

                # Clipping the probabilities [0, 1]
                probabilities = torch.nn.functional.softmax(outputs, 1)

                pos_probabilities.append(probabilities[:,1].tolist()) # To obtain only the number from the tensor
                validation_labels.append(labels.tolist()) # To obtain only the number from the tensor

                eval_acc += torch.sum(predictions == labels.data)
                eval_loss = criterion_train(outputs, labels)

        eval_acc /= folds[eval_fold]
        eval_loss /= folds[eval_fold]

        print(f'Validation set {eval_fold} accuracy = {eval_acc}')
        print(f'Validation set {eval_fold} loss = {eval_loss}')

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            print(f'Validation best accuracy updated {eval_acc}')

    # Appending the scores with their respective classes in the dataframe
    for list_probabilities, list_labels in zip(pos_probabilities, validation_labels):
        for pos_prob, label in zip(list_probabilities, list_labels):
            # Using only 4 decimal digits
            pos_prob = float("{:.4f}".format(pos_prob))

            instance = {'score': pos_prob, 'class': label}
            instance = pd.DataFrame([instance])
            val_scores = pd.concat([val_scores, instance], ignore_index=True)

    return model_cnn, val_scores



if __name__ == '__main__':
    # Device for running the train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, help='Experiment name')
    parser.add_argument('-m', '--model', default='resnet50', type=str, help='CNN model')
    parser.add_argument('-pt', '--pretrained', default=0, type=int, help='Pretrained model or not')
    args = parser.parse_args()


    experiment_name = args.experiment
    model_name = args.model
    if(args.pretrained == 0):
        pre_trained = 'NotPreTrained'
    else:
        pre_trained = 'PreTrained'

    # Getting the images and transforming it for the CNN
    image_datasets = transform_images('trains/' + experiment_name)

    folds = [660, 660]
    train_set = torch.utils.data.random_split(image_datasets, folds)

    dataloaders = {x: torch.utils.data.DataLoader(train_set[x], batch_size=16, shuffle=True, num_workers=2)
                    for x in range(len(train_set))}

    _, classes = next(iter(dataloaders[0]))

    # Setting the model
    model = setting_model(model_name)
    criterion, optimizer, scheduler = training_parameters()

    model, validation_scores = train_model(model, criterion, optimizer, scheduler, 100)

    # Saving model weights
    model_path = os.path.abspath('models/' + experiment_name + '/' + pre_trained)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Saving the validation scores into a csv for future usage
    scores_path = os.path.abspath('scores/' + experiment_name + '/' + pre_trained)
    if not os.path.exists(scores_path):
        os.makedirs(scores_path)

    validation_scores.to_csv('scores/' + experiment_name + '/' + pre_trained + '/ValidationScores_' + model_name + '_' + experiment_name + '.csv', index=False)
    torch.save(model.state_dict(), 'models/' + experiment_name + '/' + pre_trained + '/' + model_name + '_' + experiment_name + '.pth')

    print('Model for experiment ' + experiment_name + ' trained successfully')

