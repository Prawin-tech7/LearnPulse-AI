# Week 4 – Explainable AI and Model Interpretability

## Objective

The objective of Week 4 was to improve the transparency and interpretability of the machine learning model developed in Week 3. While the Random Forest model achieved good predictive performance, understanding why the model makes specific predictions is equally important in educational applications.

This phase focuses on Explainable Artificial Intelligence (XAI) techniques that help identify the factors influencing student academic outcomes and engagement predictions.

## Dataset

The project uses the processed Open University Learning Analytics Dataset (OULAD) prepared during Week 2.

The dataset contains student demographic information, assessment performance, and engagement metrics extracted from Virtual Learning Environment (VLE) interactions.

## Model Used

The optimized Random Forest Classifier developed during Week 3 was reused for explainability analysis.

Best Hyperparameters:

- n_estimators = 200
- max_depth = 20

## Explainability Techniques Implemented

### 1. Feature Importance Analysis

Feature importance scores were extracted from the Random Forest model to determine which variables contribute most to prediction outcomes.

Key influential features:

- assessment_count
- average_score
- total_clicks
- avg_clicks

### 2. Permutation Importance Analysis

Permutation Importance was used to validate feature significance by measuring performance degradation when feature values were randomly shuffled.

This provided an additional layer of confidence in the interpretation results.

## Results

The analysis revealed that academic performance and engagement-related variables have the strongest influence on student outcome predictions.

Students with higher assessment participation, stronger average scores, and greater learning platform engagement generally achieved better academic outcomes.

## Deliverables

- LearnPulse_XAI.ipynb
- LearnPulse AI - Week 4 Report.pdf
- README.md

## Conclusion

Week 4 successfully enhanced the transparency of the LearnPulse AI system by explaining how the model generates predictions. The use of Explainable AI techniques improved trust, accountability, and understanding of the predictive model, making the system more suitable for real-world educational applications.