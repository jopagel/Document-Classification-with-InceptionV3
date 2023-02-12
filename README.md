
### Document Classification with InceptionV3 

In this project, the documents of the Tobacco3482 dataset will be classified. For this purpose, 
the InceptionV2 is fine-tuned on the Tobacco2382 data,
the best hyperparameters are determined using Ray Tune, and the best model finally evaluated using a 
holdout test set of the Tobacco3482. 

*Note: There is a [blog post](https://medium.com/@jopagel/document-classification-with-inceptionv3-290e2af6628d)
about this project that goes into more detail about the procedure that was carried out.*

#### 0 Download dataset and install requirements 

In order to finetune the InceptionV3 on the Tobacco3482 dataset, you have to [download](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg) it. 
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

*Note: The helper.py file contains the functions used in the notebook.*

#### 3 Results

By performing hyperparameter tuning, a validation accuracy of ~78.1% with a batch size of 4, a learning rate of ~0.003,
a momentum of 0.6 and a weight decay of ~0.008 could be achieved. 
On the hold out test set, this model scored a test accuracy of ~81.3%. 

#### 4 Acknowledgements

- Ferrando, J., J. L. Domnguez, J. Torres, R. Garca, D. Garca, D. Garrido, J. Cortada, and
M. Valero (2020). Improving accuracy and speeding up document image classification
through parallel systems. International Conference on Computational Science – ICCS 2020.
ICCS 2020. Lecture Notes in Computer Science, vol 12138. Springer, Cham 12138, 387–400.

- Harley, A. W., A. Ufkes, and K. G. Derpanis (2015). Evaluation of deep convolutional nets
54
for document image classification and retrieval. In International Conference on Document
Analysis and Recognition (ICDAR).

- Jayant Kumar, Peng Ye and David Doermann. "Structural Similarity for Document Image Classification and Retrieval." Pattern Recognition Letters, November 2013. 

- https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg
- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
- https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html
- https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
- https://docs.ray.io/en/latest/tune/api_docs/reporters.html
- https://arxiv.org/pdf/2004.07922v1.pdf