## Predicting Apartment Prices in Beijing

### Overview
This project focuses on predicting apartment prices in Beijing using machine learning models. The dataset used for this analysis contains various features related to apartment listings, including location, amenities, and other relevant details. The goal is to build a model that accurately predicts apartment prices based on these features.

### Steps

1. **Data Cleaning:**
   - Removed irrelevant columns such as 'url', 'id', and others.
   - Eliminated outliers by trimming the dataset based on the total price.
   - Standardized column names to lowercase.

2. **Data Exploration:**
   - Utilized various visualization techniques, including scatter plots and maps, to explore the relationship between apartment prices and different features.
   - Conducted correlation analysis to identify relationships between numerical variables.
   - Split the dataset into training and testing sets using stratified sampling based on median income categories.

3. **Data Preparation:**
   - Developed a custom transformer (`MyMultiOutputLabelBinarizer`) to handle categorical data.
   - Created pipelines for numerical and categorical data transformations.
   - Employed a ColumnTransformer to combine these pipelines into a preprocessor.

4. **Model Building:**
   - Constructed baseline models using Linear Regression, Decision Tree, and Random Forest.
   - Addressed underfitting and overfitting issues by evaluating model performance and selecting Random Forest as the most promising model.
   - Saved the trained models (Linear Regression, Decision Tree, and Random Forest) for future use.

5. **Fine-Tuning Models:**
   - Conducted a grid search to find the best hyperparameters for the Random Forest model.
   - Identified the importance of each feature in the selected model.

6. **Model Evaluation:**
   - Evaluated the final model on the test dataset to assess its performance on unseen data.
   - Calculated the Root Mean Squared Error (RMSE) as a measure of prediction accuracy.

### Conclusion
The Random Forest model, after fine-tuning, demonstrates promising results in predicting apartment prices in Beijing. The project provides a comprehensive workflow from data cleaning and exploration to model building and evaluation. The saved models can be easily used for future predictions or integrated into other applications.

### Code and Resources
The complete code for this project can be found in the provided Python script. The dataset used is named 'housing_data2017.csv', and the models are saved as 'linear_reg.pkl', 'decisiontree_reg.pkl', and 'randomforest_reg.pkl'.
