#!/usr/bin/env python
# coding: utf-8

# In[187]:


# 1 Set up the environment Install Python (Anaconda Navigator, then PyCharm ), NumPy, Pandas, Seaborn, Matplotlib
''' I installed them one by one as I needed them to avoid errors.
    When I wanted to use another one I installed it and then press Run again.
'''

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from joblib import dump


# 2 Getting and loading the data
'''I'll use the Laptop Price dataset */
   From https://www.kaggle.com/datasets/muhammetvarl/laptop-price
   which I saved in my computer
'''

df = pd.read_csv('C:/Users/Julia/Desktop/LaptopPriceProject/laptop_price.csv')


# In[188]:


# 3 EDA

# EDA Summary
# 3.1 The first few rows of the dataset
# 3.2 Checking the size of the dataset
# 3.3 Getting a dataframe summary
# 3.4 Check the columns' names and rename for consistency correcting with lowercase, and without spaces
# 3.5 Checking each column's data types
# 3.6 Cleaning 'ram' column
# 3.7 Checking for min and max values
# 3.8 Checking for missing values

# 3.1 Take a look at the first few rows of the dataset
print(df.head())

# 3.1 Take a closer look at the first few rows of the dataset
''' In PyCharm, because the above command is displaying only a few rows,
    Set the display option to show all columns
    pd.set_option('display.max_columns', None)
    and then take another look at the first few rows of the dataset
    df = pd.read_csv('C:/Users/Julia/Desktop/LaptopPriceProject/laptop_price.csv')
    print(df.head(2))
'''


# In[189]:


# 3.2 Checking the size of the dataset
'''A method which provides essential information about a DataFrame including:
    The number of entries/rows.
    The number of columns.
    The column names.
    The number of non-null values in each column.
    The data type of each column.
    The memory usage of the data.
    This method is particularly useful for a quick examination of the data's structure and to assess which columns have missing data that may require cleaning.
'''
# 3.2 Checking the shape of the dataset (the numbers of rows and columns)
print(df.shape)


# In[190]:


# 3.3 Getting a dataframe summary
print(df.info())


# In[191]:


# 3.4 Now let's take a look only at the columns' names
print(df.columns)


# In[192]:


# 3.4 Rename the columns
df.rename(columns={'OpSys': 'operation_system'}, inplace=True)
df.rename(columns={'TypeName': 'type_name'}, inplace=True)
df.rename(columns={'ScreenResolution': 'screen_resolution'}, inplace=True)
df.rename(columns={'Price_euros': 'price_euros'}, inplace=True)

# and let's see the changes
print(df.columns)


# In[193]:


# 3.4 Replace uppercase letters with lowercase, and empty spaces with underlines
df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Let's see the changes with print(df.columns) or
df.head()


# In[194]:


# 3.5 Checking the data types of each column to understand what kinds of data are present
print(df.dtypes)


# In[195]:


# 3.6 Replace 'GB' with an empty string in the 'ram' column
df['ram'] = df['ram'].str.replace('GB', '')

# And take another look at the first few rows of the dataset
print(df.head())


# In[196]:


# 3.7 Checking for min and max values
'''Examining the minimum and maximum values of each feature in a dataset
    is a fundamental aspect of exploratory data analysis.
    It gives you a sense of the range of values 
    and can help identify possible outliers or errors in data collection.
'''

# Get min-max values for all numerical features
min_max_values = df.describe().loc[['min', 'max']]
print(min_max_values)


# In[197]:


# 3.7 Checking for missing values
missing_values = df.isnull().sum()
print(missing_values)


# In[198]:


# Log transform the price column.
df['log_price'] = np.log1p(df['price_euros'])

# but first make sure that numpy is installed


# In[199]:


# Creating a histogram of the 'price' column from the DataFrame 'df'
'''but first make sure that seaborb and matplotlib are installed'''

sns.histplot(df['price_euros'], bins=50, color='black', alpha=1)

'''This line above uses Seaborn's histplot function to plot a histogram.
    df['price'] specifies the data from the 'price' column in the dataframe.
    bins=50 divides the data into 50 bins for the histogram, providing a detailed view of the distribution.
    color='black sets the color of the histogram bars to black.
    alpha=1 makes the bars fully opaque.
'''


# In[200]:


# aaaand it works!
# Now let's beautify this histogram addind appropriate labels

'''
    Setting the label for the y-axis as 'Quantity'
    plt.ylabel('Quantity')
    The ylabel function is used to set the label of the y-axis.
    'Quantity' refers to the number of occurrences or frequency of prices within each bin.

    Setting the label for the x-axis as 'Price'
    plt.xlabel('Price')
    The xlabel function is used to set the label of the x-axis.
    'Price' indicates that the histogram is showing the distribution of different price values.

    Setting the title of the histogram plot
    plt.title('Distribution of prices')
    The title function is used to set the title of the plot.
    'Distribution of prices' gives a descriptive title, indicating what the plot represents.

    Displaying the plot
    plt.show()
    The show function renders the plot and displays it to the user.
    It's a necessary command to actually visualize the plot when using Matplotlib.
'''

sns.histplot(df['price_euros'], bins=50, color='black', alpha=1)
plt.ylabel('Quantity')
plt.xlabel('Price')
plt.title('Distribution of prices')
plt.show()


# In[201]:


# Validation Framework
np.random.seed(2)
n = len(df)
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)
idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]
df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()

y_train_orig = df_train.price_euros.values
y_val_orig = df_val.price_euros.values
y_test_orig = df_test.price_euros.values

y_train = np.log1p(y_train_orig)
y_val = np.log1p(y_val_orig)
y_test = np.log1p(y_test_orig)

del df_train['price_euros']
del df_val['price_euros']
del df_test['price_euros']

# Linear Regression Model Training
def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Feature Preparation
base = ['company', 'type_name', 'inches']  # Make sure these are the correct feature names

def prepare_X(df):
    df_num = df.copy()
    for col in base:
        df_num[col] = pd.to_numeric(df_num[col], errors='coerce')
    df_num = df_num.fillna(0)
    X = df_num[base].values
    return X

# Prepare the data
X_train = prepare_X(df_train)
X_val = prepare_X(df_val)

# Train the model
model = train_linear_regression(X_train, y_train)

# Save the trained model
dump(model, 'model.joblib')

# Make predictions
y_pred = model.predict(X_val)

# Plot the predictions vs actual distribution
plt.figure(figsize=(6, 4))
sns.histplot(y_val, label='target', color='#222222', alpha=0.6, bins=40)
sns.histplot(y_pred, label='prediction', color='#aaaaaa', alpha=0.8, bins=40)
plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Predictions vs actual distribution')
plt.show()


# In[202]:


from flask import Flask, request, jsonify
from joblib import load
from threading import Thread

app = Flask(__name__)

# Load the trained model
model = load('model.joblib')

@app.route('/')
def home():
    return "This is the model prediction service!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Preprocess the input data as required, similar to how you did in the notebook
    # For example, if you expect a single feature called 'feature_input'
    input_data = [data['feature_input']]
    # Use the model to make a prediction
    prediction = model.predict([input_data])
    return jsonify({'prediction': prediction.tolist()})

# Define the function that will run the Flask app
def run_app():
    # Set the threaded argument to True to handle each request in a separate thread.
    app.run(port=6969, debug=True, use_reloader=False, threaded=True)

# Run the Flask app in a separate thread to avoid blocking the notebook
flask_thread = Thread(target=run_app)
flask_thread.start()


# In[203]:


pip freeze > requirements.txt


# In[ ]:




