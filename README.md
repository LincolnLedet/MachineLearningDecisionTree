# MachineLearningDecisionTree
This project implements a machine learning model based on decision trees that processes data from CSV files to build a model
capable of making accurate predictions on new datasets. It utilizes various machine learning methods to predict the outcomes
of any given dataset.

The model works by selecting a category from the dataset and making decisions based on the median value of that category. The 
**DTLearner** selects the category most correlated with the target variable we aim to predict, while the **RTLearner** picks a
category randomly. Both **RTLearner** and **DTLearner** can be integrated into the **BagLearner** class, which creates 
multiple decision trees to produce a more robust and reliable model.

This program successfully achieved 87% accuracy in predicting heart attacks using a public Kaggle dataset: [Heart Attack 
Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset). 
The model can also be used to predict anything from weather patterns to wine ratings, provided the data is clean and of high
quality.

### Instructions to Run the Project

1. Download the necessary dependencies.
2. Run `BagLearner.py`.
3. Dataset, model configurations, and prediction setup can be customized in `BagLearner.py`. 
