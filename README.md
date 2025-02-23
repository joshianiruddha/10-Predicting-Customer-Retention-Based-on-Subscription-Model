# 10-Predicting-Customer-Retention-Based-on-Subscription-Model
Predicting Customer Retention Based on Subscription Model

## Predicting Customer Churn in Subscription Services
Business Problem
This project aims to predict customer churn for a subscription-based service and identify key factors that contribute to churn.  By predicting which customers are likely to churn, the company can implement targeted retention strategies to minimize revenue loss.    

## Data
The primary dataset is a customer churn dataset from Kaggle.    

Data Source: The link is: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/data    

Data Format: The data is provided in a CSV file format.    

Data Contents: The dataset contains up to 12 variables.    

These include:

CustomerID

Age

Gender

Tenure

Usage

Usage Frequency

Support Requests

Support Calls

Payment Delay

Subscription Type

Contract Length

Total Spend

Last Interaction

Churn    

## Data Preparation
Libraries: The project uses Python libraries such as Pandas, NumPy, and Seaborn for data cleaning, manipulation, and visualization.    

Cleaning: The data is cleaned by removing invalid or incomplete rows and filling missing data with mean or averages.    

The data types of columns were converted to appropriate types.    

Categorical Data: Categorical data is converted using one-hot encoding as required.  The 'Gender', 'Subscription Type' and 'Contract Length' columns were converted using label encoding.    

Data Exploration: The data was explored using functions to find missing data, outliers, maximum and minimum values.    

Scaling: Numerical features were scaled using StandardScaler.    

Data Dictionary: The data dictionary is provided in the "Data Contents" section above.    

It describes the type of information contained in each column.    

## Methods
Data Exploration: Pandas, NumPy and Seaborn libraries were used to clean the data by removing invalid/incomplete rows and filling missing data.    

Visualizations such as scatter plots, histograms and box plots were created using Matplotlib, Plotly and/or Seaborn to identify any relationships between variables.    

Data distributions, outliers, and missing values were explored.    

Feature Engineering: Feature engineering may not be required with the given variables.    

A cross-reference matrix review of the variables was done to ensure to identify the relationship.    

Machine Learning Modeling: Following models were considered: Logistic Regression, Random Forest, SVM, Neural Networks.    

The best model was to be evaluated.    

For neural networks the Scikit-learn libraries were used.    

The data was split into training and testing sets, with 80% used for training and 20% for testing.    

The Neural Network model was defined with input layers, hidden layers (with ReLU activation), and an output layer (sigmoid activation).    

Model Evaluation: Model performance was evaluated using metrics like accuracy and loss.    

The accuracy, precision, recall, and f1-score were calculated for various model.    

## Analysis
Data Visualization: Data visualization was used to find relationships between the variables in the dataset.    

A correlation matrix visualized correlations between numerical features.    

Pair plots were used to see distributions and relationships.    

Box plots were used to visualize the distribution of numerical variables across different categories.    

Model Performance: The Scikit-learn NN model achieved high accuracy, precision, recall and f1-score amongst other models evaluated.    

## Conclusion
The project successfully developed a model to predict customer churn using a Scikit-learn NN network.  The model's high accuracy suggests that it can be used to identify customers at risk of churning.  This allows the business to take proactive measures to retain these customers.    

Future Uses/Additional Applications
The model can be used to identify customers at risk of churning in real-time.    

The model can be used to develop targeted marketing campaigns for customer retention.    

Further analysis can be performed to find the drivers of customer churn by looking at feature importances from the models.    

The model can be extended by including new features that might improve the model. 
