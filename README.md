# assess_credit_risk
Build and evaluate several machine learning models to assess credit risk

### Summary
A large problem in properly identifying credit risk is that there is an unbalanced classification problem; the number of good "low risk" loans outnumbers the number of risky "high risk" loans. Therefore, the following sampling techniques have been performed on the dataset to train and evaluate machine learning models with unbalanced classes. 

### Random Oversampling
- Through random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.

- Results: 

  - The accuracy score is 66.0%, indicating that each observation in the testing set was predicted correctly about 66% of the time by the model. However, this number can be misleading and additional metrics need to be generated. 
    
    <img src="images/random_oversampling_score.PNG" height="45">

  - The confusion matrix is displayed below. From the confusion matrix, it can be seen that out of the loan statuses prediced to fall within "high risk", most of them were predicted incorrectly.  
    
    <img src="images/random_oversampling_cm.PNG" height="100">

  - The classification report demonstrates the discrepancies between the model predicting a "low risk" versus "high risk" loan status. Looking at the "low risk", the model has high precision (pre) and low recall (rec). This means that of the loans predicted to be "low risk", the model is reliable for classifying "low risk" loan statuses. However, there is a large number of loans that are being classified as "high risk" but are actually "low risk", as seen in the confusion matrix. 
  
    Now analyzing the "high risk" loan statuses, there is a higher recall but very poor precision. Of the loans that are actually "high risk", there is a 74% likelihood that the model will correctly predict that it is "high risk". However, out of the outcomes where the model predicts that the loan is "high risk", there is an extremely low probability that the laon is actually "high risk". 
    
    <img src="images/random_oversampling_report.PNG" height="110">

### SMOTE: Synthetic Minority Oversampling Technique
- In SMOTE, new instances are interpolated. For an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created. 

- Results: 

  - The accuracy score of the Logistic Regression model using the SMOTE oversampling technique yielded a slightly lower value of about 65%.
  
    <img src="images/SMOTE_oversampling_score.PNG" height="45">
  
  - The confusion matrix is displayed below:
  
    <img src="images/SMOTE_oversampling_cm.PNG" height="100">
  
  - The recall for the majority class ("low risk") has increased compared to using random oversampling. However, it has decreased for the minority class ("high risk"). The precision has remained the same for both the majority and minority classes. 
  
    The tradeoff is that of the loans that are actually "low risk", more are being correctly classified. On the other hand, more of the "high risk" loans are now incorrectly being identified as "low risk". 
  
    <img src="images/SMOTE_oversampling_report.PNG" height="110">
  
### Cluster Centroid Undersampling
- In cluster centroid undersampling, the algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class. 

- Results: 

  - The accuracy score for the cluster centroid undersampling technique is low at 55% approximately. 
  
    <img src="images/cc_undersampling_score.PNG" height="45">
  
  - Again, the confusion matrix demonstrates that of the loans being classified as "high risk", the majority are actually "low risk". Contrary, the "low risk" loans are correctly being identified when properly predicted.
  
    <img src="images/cc_undersampling_cm.PNG" height="100">
    
  - The recall of the majority "low risk" class is the lowest compared to the previous models. This is illustrated by the confusion matrix; the majority of the outcomes have been incorrectly predicted as "high risk".
  
    <img src="images/cc_undersampling_report.PNG" height="110">

### Combination Sampling With SMOTEENN
- SMOTEEN combines the SMOTE and ENN (Edited Nearest Neighbors) algorithms by following a two-step process. First, the minority class is oversampled with SMOTE. Then, the resulting data is cleaned with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that point is dropped. 

- Results: 

  - The accuracy score is highest at about 68% using the SMOTEEN technique. 
  
    <img src="images/SMOTEEN_combosampling_score.PNG" height="45">
  
  - Compared to the cluster centroid undersampling technique, the model has improved at properly identifying "low risk" loans. 
  
    <img src="images/SMOTEEN_combosampling_cm.PNG" height="100">
    
  - The precision of both the majority and minority class has remained the same for each of the sampling techniques. This indicates that the reliability of a positive classification for a "high risk" loan is low, while the positive classification of a "low risk" loan is high when predicted by the model. 
  
    Looking at the recall scores, there is a 59% likelihood that of the loans that are actually "low risk" are correctly identified. Also, of the actual "high risk" loans, there is a 77% likelihood that the loan will be properly classified. Of all of the sampling techniques, SMOTEEN has resulted in the highest recall for the minority "high risk" class. 
  
    <img src="images/SMOTEEN_combosampling_report.PNG" height="110">

### Conclusion

Overall, the machine learning models tested would not be recommended. The main goal of the machine learning model would be to properly identify which "high risk" loans. While the recall score of the SMOTEEN technique was the best compared to the other models, the precision is too low at 0.01. The tradeoff between the recall and the precision for the minority class is too great. Although of the actual "high risk" loans, most are being predicted correctly, too many "low risk" loans are being misclassified as "high risk".
