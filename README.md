# Indian Railways Waitlist Predictor

An end-to-end machine learning pipeline that predicts the probability of a waitlisted railway ticket clearing, using XGBoost.

## 🛠️ The Engineering Challenge
The initial dataset suffered from severe **Target Leakage** (tickets that were instantly confirmed were mixed with waitlisted tickets, resulting in a false 100% accuracy). Furthermore, the Kaggle dataset lacked real-world temporal churn. 

To solve this and build a realistic predictor, I engineered a **Synthetic Churn Simulation** that mathematically models waitlist clearances based on queue position and days to departure, effectively transforming a static dataset into a dynamic prediction challenge.

## 📊 Key Results
* **Algorithm:** XGBoost Classifier
* **Performance:** 88% Accuracy
* **Feature Engineering:** Implemented high-cardinality filters and temporal feature extraction.

## 🔍 Feature Importance
![Feature Importance](feature_importance.png)
*The model successfully proved that Waitlist Position and Days to Departure are the driving mathematical factors behind ticket confirmation.*
