# Credit-Card-Fraud-Detection

Credit Card Fraud Detection.

link of dataset=https://www.kaggle.com/mlg-ulb/creditcardfraud. The datasets contain credit card transactions over a two-day collection period in September 2013 by European cardholders. There are a total of 284,807 transactions, of which 492 (0.172%) are fraudulent.

We have used numpy, pandas, matplotlib, sklearn library for pre-processing the dataset to check whether the dataset contains balanced or imbalanced data. Since the dataset was imbalanced, we have resampled the imbalanced data by applying SMOTE. Machine Learning Algorithms like Logistic-Regression, Decision-Tree, Random-Forest have used to train the dataset. We have used Voting Classifier which dynamically selects the algorithm which gives the best precision and recall. We have developed a web app using the Flask framework, which accepts the 30 features from user input and predicts whether the given data is fraudulent or non-fraudulent.

Precision and Recall of our model: Precision - 0.99, Recall - 0.99
