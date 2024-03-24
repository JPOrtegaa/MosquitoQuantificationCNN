
import os
import pdb
import random
import math
import warnings
import argparse

import torch
from torchvision import transforms, datasets, models
import torch.utils.data

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

import sys

warnings.filterwarnings("ignore")


from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.ClassifyCount import ClassifyCount
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.ProbabilisticClassifyCount import ProbabilisticClassifyCount
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.AdjustedClassifyCount import AdjustedClassifyCount
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.ProbabilisticAdjustedClassifyCount import ProbabilisticAdjustedClassifyCount
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.X import X
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.MAX import MAX
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.MedianSweep import MedianSweep
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.T50 import T50

from QuantifiersLibrary.quantifiers.DistributionMatching.DyS import DyS
from QuantifiersLibrary.quantifiers.DistributionMatching.HDy import HDy
from QuantifiersLibrary.quantifiers.DistributionMatching.SORD import SORD

from QuantifiersLibrary.utils import Quantifier_Utils


def transform_images(image_folder):
    data_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    images_transformed = datasets.ImageFolder(os.path.join(image_folder), transform=data_transform)

    return images_transformed

def setting_model(cnn_model_name):
    if cnn_model_name == 'resnet50':
        cnn_model = models.resnet50()
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device) # Device from main (GPU or CPU)

        num_features = cnn_model.module.fc.in_features
        cnn_model.module.fc = torch.nn.Linear(num_features, 2) # 2 because it's binary classification, change it to get the numbers of folder maybe!

    elif cnn_model_name == 'vgg16':
        cnn_model = models.vgg16()
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)

        num_features = cnn_model.module.classifier[6].in_features
        cnn_model.module.classifier[6] = torch.nn.Linear(num_features, 2)
    else:
        print('Invalid model')
        sys.exit(1)

    cnn_model = cnn_model.cuda()

    return cnn_model

def run_quantifier(quant, test_scores, tprfpr, validation_scores, thr=0.5, measure='topsoe'):

    proportion_result = None

    if quant == 'CC':
        cc = ClassifyCount(classifier=None, threshold=thr)
        proportion_result = cc.get_class_proportion(scores=test_scores)

    elif quant == 'PCC':
        pcc = ProbabilisticClassifyCount(classifier=None)
        proportion_result = pcc.get_class_proportion(scores=test_scores)

    elif quant == 'ACC':
        acc = AdjustedClassifyCount(classifier=None, threshold=thr)
        acc.tprfpr = tprfpr
        proportion_result = acc.get_class_proportion(scores=test_scores)

    elif quant == 'PACC':
        pacc = ProbabilisticAdjustedClassifyCount(classifier=None, threshold=thr)
        pacc.tprfpr = tprfpr
        proportion_result = pacc.get_class_proportion(scores=test_scores)

    elif quant == 'X':
        x = X(classifier=None)
        x.tprfpr = tprfpr
        proportion_result = x.get_class_proportion(scores=test_scores)

    elif quant == 'MAX':
        quant_max = MAX(classifier=None)
        quant_max.tprfpr = tprfpr
        proportion_result = quant_max.get_class_proportion(scores=test_scores)

    elif quant == 'MS':
        ms = MedianSweep(classifier=None)
        ms.tprfpr = tprfpr
        proportion_result = ms.get_class_proportion(scores=test_scores)

    elif quant == 'T50':
        t50 = T50(classifier=None)
        t50.tprfpr = tprfpr
        proportion_result = t50.get_class_proportion(scores=test_scores)

    elif quant == 'DyS':
        dys = DyS(classifier=None, similarity_measure=measure, data_split=None)

        dys.p_scores = validation_scores[validation_scores['class'] == 1]
        dys.p_scores = dys.p_scores['score'].tolist()

        dys.n_scores = validation_scores[validation_scores['class'] == 0]
        dys.n_scores = dys.n_scores['score'].tolist()

        dys.test_scores = [score[1] for score in test_scores]

        proportion_result = dys.get_class_proportion()

    elif quant == 'HDy':
        hdy = HDy(classifier=None, data_split=None)

        hdy.p_scores = validation_scores[validation_scores['class'] == 1]
        hdy.p_scores = hdy.p_scores['score'].tolist()

        hdy.n_scores = validation_scores[validation_scores['class'] == 0]
        hdy.n_scores = hdy.n_scores['score'].tolist()

        hdy.test_scores = [score[1] for score in test_scores]

        proportion_result = hdy.get_class_proportion()

    elif quant == 'SORD':
        sord = SORD(classifier=None, data_split=None)

        sord.p_scores = validation_scores[validation_scores['class'] == 1]
        sord.p_scores = sord.p_scores['score'].tolist()

        sord.n_scores = validation_scores[validation_scores['class'] == 0]
        sord.n_scores = sord.n_scores['score'].tolist()

        sord.test_scores = [score[1] for score in test_scores]

        proportion_result = sord.get_class_proportion()


    return proportion_result

def get_best_threshold(pos_prop, pos_scores, thr=0.5, tolerance=0.01):
    min = 0.0
    max = 1.0
    max_iteration = math.ceil(math.log2(len(pos_scores))) * 2 + 10
    for _ in range(max_iteration):
        new_proportion = sum(1 for score in pos_scores if score >= thr) / len(pos_scores)

        if abs(new_proportion - pos_prop) < tolerance:
            return thr

        elif new_proportion > pos_prop:
            min = thr
            thr = (thr + max) / 2

        else:
            max = thr
            thr = (thr + min) / 2

    return thr

def classifier_accuracy(pos_proportion, pos_test_scores, labels):
    sorted_scores = sorted(pos_test_scores)

    threshold = get_best_threshold(pos_proportion, sorted_scores)

    pred_labels = [1 if score >= threshold else 0 for score in pos_test_scores]

    corrects = sum(1 for a, b in zip(pred_labels, labels) if a == b)
    accuracy = round(corrects / len(pred_labels), 2)

    return accuracy, threshold

def run_fscore(pos_probs, label_list, thr):
    pred = [1 if prob >= thr else 0 for prob in pos_probs]
    fscore = f1_score(label_list, pred)
    return fscore

def experiment(model, dts_images, val_scores, model_name, boolean_pt):

    columns = ['sample', 'test_size', 'alpha', 'actual_prop', 'pred_prop', 'abs_error', 'acc', 'f_score', 'thr',
               'quantifier', 'model', 'pre_trained']
    result_table = pd.DataFrame(columns=columns)

    pos_index = [i for i in range(len(dts_images)) if dts_images[i][1] == 1]
    neg_index = [i for i in range(len(dts_images)) if dts_images[i][1] == 0]

    pos_dataset = torch.utils.data.Subset(dts_images, pos_index)
    neg_dataset = torch.utils.data.Subset(dts_images, neg_index)

    quantifiers = ['CC', 'ACC', 'X', 'MAX', 'T50', 'MS', 'DyS', 'HDy', 'SORD']

    test_sizes = [10, 20, 30, 40, 50, 100]
    alpha_values = [round(x, 2) for x in np.linspace(0, 1, 21)]
    print(alpha_values)
    iterations = 10

    arrayTPRFPR = Quantifier_Utils.TPRandFPR(validation_scores=val_scores)

    for size in test_sizes:
        for alpha in alpha_values:
            for iteration in range(iterations):
                pos_size = int(np.round(size * alpha, 2))
                pos_index = random.sample(range(len(pos_dataset)), pos_size)

                neg_size = size - pos_size
                neg_index = random.sample(range(len(neg_dataset)), neg_size)

                pos_subset = torch.utils.data.Subset(pos_dataset, pos_index)
                neg_subset = torch.utils.data.Subset(neg_dataset, neg_index)

                test_dataset = torch.utils.data.ConcatDataset([pos_subset, neg_subset])
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, num_workers=2, shuffle=True)

                pos_proportion = round(pos_size/size, 2)

                model.eval()
                with torch.no_grad():
                    probabilities = []
                    labels_list = []
                    for inputs, labels in test_dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        new_probabilities = torch.nn.functional.softmax(outputs, 1)

                        # Transforming to list and using only 4 decimals
                        new_probabilities = new_probabilities.tolist()
                        new_probabilities = [[round(prob, 4) for prob in row] for row in new_probabilities]

                        labels = labels.tolist()

                        labels_list.append(labels)
                        probabilities.append(new_probabilities)

                    # Transforming in only a list of probabilities (2 dimensions)
                    probabilities = [prob for prob_list in probabilities for prob in prob_list]
                    labels_list = [lab for lab_list in labels_list for lab in lab_list]

                    pos_probabilities = [prob[1] for prob in probabilities]

                    for quantifier in quantifiers:
                        predicted_proportion = run_quantifier(quantifier, probabilities, arrayTPRFPR, val_scores) # Added val_scores
                        pos_prediction = predicted_proportion[1]

                        accuracy, thr = classifier_accuracy(pos_prediction, pos_probabilities, labels_list)

                        abs_error = round(abs(pos_prediction - pos_proportion), 2)

                        f_score = run_fscore(pos_probabilities, labels_list, thr)

                        instance = {'sample': iteration+1, 'test_size': size, 'alpha': alpha,
                                    'actual_prop': pos_proportion, 'pred_prop': pos_prediction,
                                    'abs_error': abs_error, 'acc': accuracy, 'f_score': f_score, 'thr': thr,
                                    'quantifier': quantifier, 'model': model_name, 'pre_trained': boolean_pt}

                        instance = pd.DataFrame([instance])
                        result_table = pd.concat([result_table, instance], ignore_index=True)

                        # print(result_table)

    return result_table

def run_experiment(exp_name, model, pt):
    experiment_name = exp_name
    model_name = model
    pre_trained = pt

    if pre_trained == 'PreTrained':
        boolean_pt = 1
    else:
        boolean_pt = 0

    dataset_images = transform_images('tests/' + experiment_name)

    state_dict = torch.load('models/' + experiment_name + '/' + pre_trained + '/' + model_name + '_' + experiment_name + '.pth')
    model_cnn = setting_model(model_name)
    model_cnn.load_state_dict(state_dict)

    validation_scores = pd.read_csv('scores/' + experiment_name + '/' + pre_trained + '/ValidationScores_' + model_name + '_' + experiment_name + '.csv')

    result = experiment(model_cnn, dataset_images, validation_scores, model_name, boolean_pt) # Added model_name as a parameter!

    result_path = os.path.abspath('results/' + experiment_name + '/' + pre_trained)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    result.to_csv(result_path + '/ResultTable_' + model_name + '_' + experiment_name + '_Final' + '.csv', index=False) # Added _Final to the csv file!



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, help='Experiment name')
    parser.add_argument('-m', '--model', default='resnet50', type=str, help='CNN model')
    parser.add_argument('-pt', '--pretrained', default=0, type=int, help='Pretrained model or not')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    
    exp_name = args.experiment
    model_name = args.model
    if(args.pretrained == 0):
        pt_name = 'NotPreTrained'
    else:
        pt_name = 'PreTrained'

    run_experiment(exp_name, model_name, pt_name)
    print('Experiment ', exp_name, model_name, pt_name, 'concluded!')
