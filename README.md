# Serverless-ML
## Titanic Features
In our titanic-feature-pipeline.py, we retrieve the original dataset from the provided url in the assignment, then we performe data cleaning by removing useless data, as well as using one hot encoding on some of the features. Once we have our cleaned dataset, we send it to our Feature Groups on Hopsworks.

## Titanic Synthetic Features
In our titanic-feature-pipeline-daily.py, we create artifical data for our titanic dataset, that we then send to our originally created dataset on Hopworks. This newly created data gets appended to the last row of our original dataset. 

## Titanic Training
In our titanic-training-pipeline.py, we first retrieve our training data from Hopsworks, split it in train/test groups and use an XGBClassifier on it. At best, our model managed to get a testing accuracy of 88%, depending on the initial parameters used. Once our model created, we send it to our model registry on Hopsworks. 

## Titanic Latest Predictions
In our titanic-batch-inference-pipeline.py, we create a schedule that runs at a given interval. This means that at each time interval, the model makes a prediction on the whole dataset. What we are interested in however, is the prediction of the latest created synthetic data, i.e the last row of the dataset. Once we have the latest prediction, we create the updated confusion matrix, all of which are then saved to be uploaded on the huggingface space. 

## Huggingface Spaces
Our last task required us to create two huggingface spaces, one interactive UI which enabled the user to enter various features for a passenger and see whether the passenger survived or not, and one which simply renders the latest results obtained from our synthetic data. The spaces can be found at the following addresses: <br>
https://huggingface.co/spaces/ayberkuckun/titanic <br>
https://huggingface.co/spaces/ayberkuckun/titanic-monitoring
