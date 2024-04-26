# MosquitoQuantificationCNN_ERBD24

## Description
Repository referent to ERBD2024 paper Quantificação de mosquitos _Aedes aegypti_ a partir de imagens de _smartphones_

## Dataset and CNN model example
Datasets used for CNN training and testing on link below
* [Train/Test images](https://drive.google.com/file/d/1y5ruGnCb3G9-xhcAeGtTkcL3AT1p-N-d/view?usp=sharing) (635.7 MB)

Trained CNN model example on link below
* [Model](https://drive.google.com/file/d/1EKLwfQE5wG7MPkPSV-4JW_X4cetaLH4l/view?usp=sharing) (474.9 MB)

Note: datasets that were not created by COEN have _AuthorName in the ending of the folder

## Installation
```plaintext
git clone https://github.com/JPOrtegaa/MosquitoQuantificationCNN_ERBD24
cd MosquitoQuantificationCNN
git clone https://github.com/JPOrtegaa/QuantifiersLibrary
pip install -r requirements.txt
```

## Usage
```plaintext
python train_CNN.py -e ExperimentName -m ModelName -pt 0
python test_CNN.py -e ExperimentName -m ModelName -pt 0
```
Where:
* e: Experiment name, a string, the name of your experiment to create the respective folders to it
* m: Model name, a string, the name of CNN model for the experiment (resnet50, vgg16)
* pt: Pretrained, an integer (0 or 1), 0 for not usage of pretrained weights in the model, 1 for usage

The analysis of the results obtained after running the experiments can be seen in the notebook file ```result_analysis.ipynb```
