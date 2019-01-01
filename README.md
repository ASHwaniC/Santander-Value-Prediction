# Santander-Value-Prediction

# Problem Statement

We need to identify the value of transactions for each potential customer.

In this, I preprocessed the data and feed the data to gradient boosting tree models.

# Workflow:

1) *Data preprocessing*- The purpose of data preprocessing is to achieve higher time/space efficiency. What we did includes round, constant features removal, duplicate features removal, insignificant features removal, etc. The key here is to ensure the preprocessing shall not hurt the accuracy.
2) *Feature transform*- The purpose of feature transform is to help the models to better grasp the information in the data, and fight overfitting. What we did includes dropping features which "live" on different distributions on training/testing set, adding statistical features, adding low-dimensional representation as features.
3) *Modeling*- We used 2 models: xgboost and lightgbm. We averaged the 2 models for the final prediction.
