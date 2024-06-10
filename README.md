
# Singapore Flat Resale Price Prediction

The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.




## Requirements
NumPy

Pandas

Scikit-learn

Streamlit
## Steps involved
1. Data Collection and Preprocessing: Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date. Preprocess the data to clean and structure it for machine learning.
2. Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. 
3. Choose an appropriate machine learning model for regression. 
4. Train the model on the historical data. 
5. Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.
6. Develop a user-friendly web application using Streamlit to utilize the trained machine learning model to predict the resale price based on user inputs.
7. Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.


## Getting started
Data source:https://beta.data.gov.sg/collections/189/view

1. Start with data preprocessing steps(main.ipynb) using the dataset provided, and store the cleaned data seperately.
2. Encode categorical data and scale the numerical data before training the models.
3. Split the data into training and testing data.
4. Choose the best regression by comparing the evaluation metrics of each model.Here  LinearRegression, DecisionTreeRegressor, RandomForestRegressor and XGBRegressor were compared and RandomForestRegressor was found to be better.But for deployment purposes Decision tree model is used.
5.  Use pickle module to dump and load models(encoder.pkl,scaler_features.pkl,scaler_target.pkl,decision_tree_model.pkl).
6. Create streamlit app to get user input and predict flat resale price.
7.  Deploy the Streamlit application to make it accessible to users over the internet.
## Skill takeaway
Data Wrangling, EDA, Model Building, Model Deployment