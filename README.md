ToDos:

- Refactoring der Funktionen aus dem Notebook ins helper.py modul und docstrings (dabei besonders auf argumente achten -> einige passen noch nicht)
- nochmal über den blog post gehen
  - config file erstellen
- gesamten neuen workflow auf colab vertesten

### Document Classification with InceptionV3 

#### 0 Download dataset and install requirements 

In order to finetune the InceptionV3 on the Tobacco 3482 dataset, you have to [download](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg) it. 
Furthermore, the required dependencies need to be installed via

```
pip install -r requirements.txt 
```

#### 1 Specify the config parameters 

Before the training, the config parameters in the config.yaml have to be specified:

- IMAGE_DIR : directory of the downloaded Tobacco3482
- MODEL_NAME : the CNN to be trained 
- NUM_CLASSES : the number of output classes (10 for the tobacco3482 dataset)
- NUM_EPOCHS : the number of epochs to train the model
- VAL_SPLIT_REL : the relative split of the validation data 
- TEST_SPLIT_REL : the relative split of the test data
- INPUT_SIZE: the image size 

#### 2 Train and tune InceptionV3

After that, the cells of the Jupyter Notebook (document_classifier_tuning.ipynb) can be executed one after the other. 