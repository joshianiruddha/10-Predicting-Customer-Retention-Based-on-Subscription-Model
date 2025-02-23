# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Project Topic:** Predicting customer retention based on subscription model
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Business Problem:** This project aims to predict customer churn for a subscription-based service and identify key factors contributing to churn, enabling targeted retention strategies to minimize revenue loss.  Customer churn occurs when customers end their relationship or subscription with a company. It measures how many customers stop using a company's products or services over time and affects revenue, growth, and customer retention.
# MAGIC
# MAGIC Customer churn can have significant impacts on a business's financial health and market position. High churn rates may indicate underlying issues such as poor customer service, product dissatisfaction, high competition, incorrect pricing or ineffective marketing strategies. 
# MAGIC
# MAGIC By addressing customer churn proactively, companies can improve their customer retention rates, maximize lifetime customer value, and ensure long-term sustainable growth.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Import Libraries
# MAGIC Import and install the required libraries

# COMMAND ----------

pip install tensorflow

# COMMAND ----------

pip install tabulate

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import numpy as np


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier # Scikit-learn NN
from tensorflow import keras # Keras NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



import matplotlib.pyplot as plt
import seaborn as sns


import warnings

import plotly.express as px

#import tensorflow as tf
#from tensorflow.keras import Sequential  
#from tensorflow.keras.layers import Dense, Dropout, InputLayer
#from tensorflow.keras.optimizers import Adam  
from tabulate import tabulate

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Load the Data

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/customer_churn_dataset_training_master-1.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df.head(10))

# COMMAND ----------

# Spark DataFrame
pandas_df = df.toPandas() 

# Pandas DataFrame (pandas_df)
(pandas_df.head(3))

# COMMAND ----------

pandas_df['Churn'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Data Exploration
# MAGIC
# MAGIC Fing the total quanity of the data, missing data, na values, outliers, maximum, minimum

# COMMAND ----------

#Convert the data to appropriate datatype 
pandas_df['CustomerID'] = pandas_df['CustomerID'].astype('string')
pandas_df['Age'] = pandas_df['Age'].astype(float)
pandas_df['Gender'] = pandas_df['Gender'].astype('category')
pandas_df['Tenure'] = pandas_df['Tenure'].astype(float)
pandas_df['Usage Frequency'] = pandas_df['Usage Frequency'].astype(float)
pandas_df['Support Calls']=pandas_df['Support Calls'].astype(float)
pandas_df['Payment Delay']=pandas_df['Payment Delay'].astype(float)
pandas_df['Subscription Type']=pandas_df['Subscription Type'].astype('category')
pandas_df['Contract Length']=pandas_df['Contract Length'].astype('category')
pandas_df['Total Spend']=pandas_df['Total Spend'].astype(float)
pandas_df['Last Interaction']=pandas_df['Last Interaction'].astype(float)
pandas_df['Churn']=np.where(pandas_df['Churn'].str.lower() == '1', True, False)


# COMMAND ----------

pandas_df.info()

# COMMAND ----------

def explore_data(df):
    results = []
    for col in df.columns:
        # Count missing values (NaN)
        missing_count = df[col].isnull().sum()
        # Count NA values (as a string)
        na_count = df[col].astype(str).str.contains('NA').sum()
        # Count total values
        total_count = len(df[col])
        # Calculate percentage of missing and NA values
        missing_percent = (missing_count / total_count) * 100
        na_percent = (na_count / total_count) * 100
        # Determine data type
        data_type = df[col].dtype

# --- Additional Checks ---

        # Unique values
        unique_count = df[col].nunique()

        # Detect outliers (using IQR method)
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):  # Only for numeric
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        else:
            outlier_count = np.nan  # Not applicable for non-numeric

        # Minimum and Maximum values
        if pd.api.types.is_numeric_dtype(df[col]):
            minimum = df[col].min()
            maximum = df[col].max()
            skewness = df[col].skew()
        else:
            minimum = np.nan
            maximum = np.nan
            skewness = np.nan
            
        results.append([
            col, total_count, missing_count, missing_percent, 
            na_count, na_percent, data_type, unique_count, 
            outlier_count, minimum, maximum, skewness  
        ])

    exploration_df = pd.DataFrame(results, columns=[
        "Column", "Total Count", "Missing Count", "Missing (%)", 
        "NA Count", "NA (%)", "Data Type", "Unique Values", 
        "Outlier Count", "Minimum", "Maximum", "Skewness"
    ])
    return exploration_df

# Perform data exploration
exploration_results = explore_data(pandas_df)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
exploration_results.head(100)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Data Preparation
# MAGIC
# MAGIC Selecting below columns for anlaysis, removing columns with high number of NAs:
# MAGIC

# COMMAND ----------

select_columns = ['Age','Gender','Tenure','Usage Frequency','Support Calls','Payment Delay','Subscription Type','Contract Length','Total Spend', 'Last Interaction'	]

data_df = pandas_df[select_columns]

data_df.describe()

# COMMAND ----------

# Drop NA records
#data_df.replace(['NA', 'Na'], pd.NA, inplace=True)
#data_df['Thermal comfort'].replace('Na', pd.NA, inplace=True)
data_df = data_df.dropna(how='any')

# COMMAND ----------

#Check data after dropping NAs
exploration_results = explore_data(data_df)
exploration_results.head(100)

# COMMAND ----------

#describe the data
data_df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Visualize the Data

# COMMAND ----------

# MAGIC %md
# MAGIC **Correlation Matrix:** Correlations between numerical features.

# COMMAND ----------

# Calculate the correlation matrix
corr_matrix = data_df.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Thermal Comfort Features')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Pair Plot:** To see both distributions of individual variables and their relationships

# COMMAND ----------

sns.pairplot(data_df)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC **Box Plot**
# MAGIC  Distribution of a numerical variable across different categories.

# COMMAND ----------

plt.figure(figsize=(20, 10))
sns.boxplot(x='Subscription Type', y='Support Calls', data=data_df)
plt.xlabel('Age')
plt.ylabel('Support Calls')
plt.title('Air Temperature by Season')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Violin Plots Density of the data distribution.

# COMMAND ----------

plt.figure(figsize=(20, 10))
sns.violinplot(x='Subscription Type', y='Total Spend', data=data_df)
plt.xlabel('Subscription Type')
plt.ylabel('Subscription Type')
plt.title('Total Spend')
plt.show()

# COMMAND ----------

#plt.figure(figsize=(20, 10))
#sns.countplot(x='Koppen climate classification', data=data_df)
#plt.xlabel('Koppen Climate Classification')
#plt.ylabel('Count')
#plt.title('Frequency of Koppen Climate Classifications')
#plt.xticks(rotation=45)  # Rotate x-axis labels if needed
#plt.show()

# COMMAND ----------

#**World map** Country Vs Child mortality

#plt.figure(figsize=(20, 10))
## Create the world map
#fig = px.choropleth(data_df,
#                    locations='country',
#                    locationmode='country names',
#                    color='child_mort',  # Choose the column you want to visualize
#                    hover_data=['gdpp', 'life_expec', 'total_fer'],
#                    color_continuous_scale='Viridis',  # Choose a color scale
#                    title='Child Mortality Rate Around the World',
#                    width=1200, 
#                    height=800)

#fig.show()

# COMMAND ----------

#**World Map** Country Vs GDP
# Create the world map
#fig = px.choropleth(data_df,
#                    locations='country',
#                    locationmode='country names',
#                    color='gdpp',  
#                    hover_data=['gdpp', 'life_expec', 'total_fer'],
#                    color_continuous_scale='Viridis',  
#                    title='GDP around the World',
#                    width=1200, 
#                    height=800)

#fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Model Training

# COMMAND ----------

pandas_df.size

# COMMAND ----------

datatrain__df.size

# COMMAND ----------

pandas_df.head()

# COMMAND ----------

pandas_df['Churn'].value_counts()

# COMMAND ----------

datatrain__df= pandas_df.copy()
label = LabelEncoder()
datatrain__df['Gender'] = label.fit_transform(datatrain__df['Gender'])
datatrain__df['Subscription Type'] = label.fit_transform(datatrain__df['Subscription Type'])
datatrain__df['Contract Length'] = label.fit_transform(datatrain__df['Contract Length'])
(datatrain__df.head())

# COMMAND ----------

print(pd.unique(datatrain__df['Churn']))
print(len(pd.unique(datatrain__df['Churn'])))

# COMMAND ----------

datatrain__df.columns

# COMMAND ----------

column_to_scale = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
scaler = StandardScaler()
for colm in column_to_scale:
    datatrain__df[colm] = scaler.fit_transform(datatrain__df[colm].values.reshape(-1,1))

# COMMAND ----------

datatrain__df.describe()

# COMMAND ----------

datatrain__df.info()

# COMMAND ----------

print(pd.unique(datatrain__df['Churn']))
print(len(pd.unique(datatrain__df['Churn'])))

# COMMAND ----------

datatrain__df.isnull().sum()

# COMMAND ----------

datatrain__df.shape

# COMMAND ----------

datatrain__df['Churn'].value_counts()

# COMMAND ----------

datatrain__df= datatrain__df.dropna()

# COMMAND ----------

datatrain__df.shape

# COMMAND ----------

print(pd.unique(datatrain__df['Churn']))
print(len(pd.unique(datatrain__df['Churn'])))

# COMMAND ----------

# Define features (X) and target (y)
X = datatrain__df.drop(['Churn','CustomerID'], axis=1)  # All columns except 'Churn'
y = datatrain__df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)  # 80% train, 20% test


# COMMAND ----------

print(pd.unique(pandas_df['Churn']))
print(len(pd.unique(pandas_df['Churn'])))

# COMMAND ----------

#Selecting different models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Scikit-learn NN": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42), # Added Scikit Learn NN
    "Keras NN": None  # We'll define this separately
}


# COMMAND ----------

# Keras NN Model Definition
keras_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # Input layer, needs input dimensions
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (sigmoid for binary classification)
])

# COMMAND ----------

keras_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
models["Keras NN"] = keras_nn

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC #### 7. Evaluation Matrix

# COMMAND ----------

for name, model in models.items():
    print(f"Training and evaluating {name}...")

    if name == "Keras NN":
      model.fit(X_train, y_train, epochs=50, batch_size=32, verbose = 0) # Train Keras model
      y_pred_probs = model.predict(X_test) # Get probabilities
      y_pred = (y_pred_probs > 0.5).astype(int) # Convert probabilities to classes (0 or 1)
    else:
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

# COMMAND ----------

data_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7. Evaluation Matrix

# COMMAND ----------


# Assuming Y_test and Y_pred are your true and predicted labels respectively

#results =[]
#for k in range(2, 11):  # Example range
#    kmeans = KMeans(n_clusters=k, random_state=48)
#    kmeans.fit(datatrain__df[column_to_scale])
    
#    inertia = kmeans.inertia_
#    labels = kmeans.labels_
#    silhouette_avg = silhouette_score(datatrain__df[column_to_scale], labels)
    
#    results.append([k, inertia, silhouette_avg])
# Accuracy
#accuracy = accuracy_score(Y_test, Y_pred)
#print(f"Accuracy: {accuracy}")

# Precision
#precision = precision_score(Y_test, Y_pred, average='macro')  # or 'micro', 'weighted'
#print(f"Precision: {precision}")

# Recall
#recall = recall_score(Y_test, Y_pred, average='macro')  # or 'micro', 'weighted'
#print(f"Recall: {recall}")

# F1-score
#f1 = f1_score(Y_test, Y_pred, average='macro')  # or 'micro', 'weighted'
#print(f"F1-score: {f1}")

# Confusion Matrix
#cm = confusion_matrix(Y_test, Y_pred)
#print("Confusion Matrix:")
#print(cm)

# Classification Report
#cr = classification_report(Y_test, Y_pred)
#print("Classification Report:")
#print(cr)

# COMMAND ----------

# Print the results in a table
#headers = ["k", "Inertia", "Silhouette Score"]
#print(tabulate(results, headers=headers, tablefmt="fancy_grid"))  # Use "fancy_grid" for a grid-like table

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see from the results, K=4 provides best results. At K=4, Silhouette Score is maximum and the slope on elbo drops down significantly.
