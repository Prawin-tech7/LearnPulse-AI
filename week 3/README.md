# Week 3 - Building and Tuning an AI Model

## Objective

The objective of this phase was to build, train, evaluate, and optimize a machine learning model capable of predicting student academic outcomes using the processed dataset generated during Week 2.

## Dataset

The model was trained using the processed LearnPulse-AI dataset created from the Open University Learning Analytics Dataset (OULAD).

## Machine Learning Algorithm

Random Forest Classifier was selected because:

- Handles mixed data types effectively
- Reduces overfitting through ensemble learning
- Provides feature importance analysis
- Performs well on educational datasets

## Tasks Performed

- Loaded processed dataset
- Selected features and target variable
- Split dataset into training and testing sets
- Trained baseline Random Forest model
- Evaluated model performance
- Performed hyperparameter tuning using GridSearchCV
- Generated confusion matrix visualization
- Analyzed feature importance

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Outcome

The tuned Random Forest model demonstrated strong predictive performance and identified important factors influencing student success and engagement.

## Files

- LearnPulse_Model_Training.ipynb
- LearnPulse_Model_Report.pdf