# Diamonds' Price Prediction
Prediction of diamond price  by their cut, color, clarity, price, and other attributes

## Introduction
This is a project assigned during the Deep Learning with Python MUP-AI05 course, one of the course for AI05 bootcamp organized by Selangor Human Resource Development Centre (SHRDC) on 13 April 2022. The data used in this project is obtained from Diamonds dataset (link: [https://www.kaggle.com/datasets/shivam2503/diamonds](https://www.kaggle.com/datasets/shivam2503/diamonds) ). The data is originally in CSV format, this file is included in this repository, you can check the file from the file list.

## Methodology
In this project, the neural network model is built with TensorFlow Keras functional API framework. Modules used in this project include:
* numpy
* pandas
* tensorflow
* matplotlib.pyplot
* sklearn.model_selection.train_test_split
* sklearn.preprocessing

## STEP 1: Prepare that data
Numpy and pandas imported in order to extract data from the csv file. After import the data, the data was inspected with pandas.DataFrame.info method in order to find out is there any missing value, also to confirm the data types. After inspection, there are 3 columns with data type of string.

## STEP 2: Split the data into features and label
The 3 columns which contain data with data type of string. These 3 column were encoded with one-hot encoding using pandas.get_dummies method. After one-hot encoding, the total number of column of the dataframe increased from 11 to 28.

Column "price" choosed to be the label as the objective of this project is to predict the diamonds' price. Since the price is a set of continuous value, this type of problem this model trying to solve is a regression problem. After drop the "price" and the "id" column from the dataframe, the rest of columns were used as features.

## STEP 3: Train Test Split and Standardization
In this step, the train and test data prepared using train_test_split method from sklearn.model_selection module. The standardization of data is done by using StandardScaler method from sklearn.preprocessing module.

## STEP 4: Design of neural network
 
#### The Neural Network model as following table
| Layer        | Output Shape | Activation Function | Param # |
|--------------|--------------|---------------------|---------|
| Input Layer  | [(None, 26)] |                     | 0       |
| Dense Layer  | (None, 64)   | relu                | 1728    |
| Dense Layer  | (None, 32)   | relu                | 2080    |
| Dense Layer  | (None, 16)   | relu                | 528     |
| Dense Layer  | (None, 8)    | relu                | 136     |
| Dense Layer  | (None, 4)    | relu                | 36      |
| Dense Layer  | (None, 2)    | relu                | 10      |
| Output Layer | (None, 1)    | relu                | 3       |

## STEP 5: Training of model
The model trained by using batch size of 32 with the epoch of 128. EarlyStopping with a patience of 5 added to the training, which cause the training stopped at epoch number 44/128.

## Result
#### The actual value of test data vs prediction value:
![Actual vs Prediction!](/actual_vs_prediction.png "Actual vs Prediction")
