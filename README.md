# Credit Card Fraud Detection using Machine Learning

This project implements a complete Machine Learning workflow to predict fraudulent credit card transactions using classification algorithms.

## Dataset
- **creditcard.csv**: Contains transaction features ($X$) and a target label ($y$) where:
  - `0` = Not Fraud
  - `1` = Fraud

## Models Implemented
- **Multinomial Naive Bayes** (requires feature scaling using `MinMaxScaler`)
- **Decision Tree Classifier**
- **Random Forest Classifier**

## Workflow Steps
1. **Load Data**: Import dataset and separate features ($X$) and target ($y$).
2. **Feature Scaling**: Scale features to $[0, 1]$ range using `MinMaxScaler`.
3. **Train-Test Split**: Split data using `train_test_split` (test size = 20%, stratified by $y$).
4. **Model Training**: Train Multinomial Naive Bayes, Decision Tree, and Random Forest models.
5. **Evaluation**: Compare models based on Accuracy.

## Results
- Multinomial Naive Bayes Accuracy: ~99.83%
- Decision Tree Accuracy: ~99.91%
- Random Forest Accuracy: ~99.96%

## Learning Outcomes
- Handling imbalanced datasets in classification tasks.
- Comparing the performance of Naive Bayes, Decision Trees, and Random Forests.
- Practicing the end-to-end ML pipeline.
