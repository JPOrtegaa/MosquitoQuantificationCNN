# MosquitoQuantificationCNN

## Description
Repository referent to ERBD2024 paper Quantificação de mosquitos Aedes aegypti a partir de imagens de smartphones


## Installation
```plaintext
git clone https://github.com/JPOrtegaa/MosquitoQuantificationCNN
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
