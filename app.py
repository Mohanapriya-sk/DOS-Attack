from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Hard-coded username and password
USERNAME = "admin"
PASSWORD = "admin"

import pandas as pd
import numpy as np

# For ploting the graphs
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine learning Model 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Machine learning model evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import base64
import io
import os


@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username == USERNAME and password == PASSWORD:
        # Redirect to dashboard on successful login
        return redirect(url_for('dashboard'))
    else:
        error = "Invalid username or password. Please try again."
        return render_template('login.html', error=error)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


def load_data():
    # Load your CSV data into a DataFrame (example)
    df = pd.read_csv("data/DDos.csv")
    return df

def load_data2():
    # Load your CSV data into a DataFrame (example)
    df = pd.read_csv("data/MalwareData.csv", sep="|", low_memory =True )
    return df

@app.route('/dos')
def dos():
    # Call load_data() to get the DataFrame
    df = load_data()    
    # Get the first 15 rows of the DataFrame
    df_first_15 = df.head(15)
    # Convert the first 15 rows DataFrame to HTML table
    html_table = df_first_15.to_html(classes='data', header="true", index=False)
    
    # Show the plot
#    plt.show()
    return render_template('dos.html', table=html_table, plot_null_filename='static/img/null_plot.png', plot_miss_filename='static/img/miss_plot.png')

@app.route('/malware')
def malware():
    # Call load_data() to get the DataFrame
    df = load_data2()    
    # Get the first 15 rows of the DataFrame
    df_first_15 = df.head(15)
    # Convert the first 15 rows DataFrame to HTML table
    html_table = df_first_15.to_html(classes='data', header="true", index=False)
    
    # Show the plot
#    plt.show()
    return render_template('malware.html', table=html_table, plot_null_filename='static/img/mal/null_plot.png', plot_miss_filename='static/img/mal/miss_plot.png')

@app.route('/processdos')
def processdos():
    # Call load_data() to get the DataFrame
    df = load_data()
    data_f=df.dropna()    
    # Get the first 15 rows of the DataFrame
    df_first_15 = data_f.head(15)

    # Convert the first 15 rows DataFrame to HTML table
    html_table = df_first_15.to_html(classes='data', header="true", index=False)
    
    # Show the plot
#    plt.show()
    return render_template('processdos.html', table=html_table, plot_remove_filename='static/img/removenull_plot.png',plot_stat_filename='static/img/benignddos_plot.png')

@app.route('/processmal')
def processmal():
    # Call load_data() to get the DataFrame
    df = load_data2()
    data_f=df.dropna()    
    # Get the first 15 rows of the DataFrame
    df_first_15 = data_f.head(15)

    # Convert the first 15 rows DataFrame to HTML table
    html_table = df_first_15.to_html(classes='data', header="true", index=False)
    
    # Show the plot
#    plt.show()
    return render_template('processmal.html', table=html_table, plot_remove_filename='static/img/mal/removenull_plot.png',plot_stat_filename='static/img/mal/benignddos_plot.png')


@app.route('/benignddos')
def benignddos():
    # Call load_data() to get the DataFrame
    df = load_data()
    df.columns = df.columns.str.strip()
    clean_df=df.dropna() 
     # Map labels to numerical values (assuming 'Label' column contains categorical labels)
    clean_df.loc[:, 'Label'] = clean_df['Label'].map({'BENIGN': 0, 'DDoS': 1})
    # data_f['Label'] = data_f['Label'].map({'BENIGN': 0, 'DDoS': 1})
    data_f1=clean_df.describe() 
    # Get the first 15 rows of the DataFrame
    # df_first_15 = data_f1.head(15)

    # Convert the first 15 rows DataFrame to HTML table
    html_table = data_f1.to_html(classes='data', header="true", index=False)
    plots_directory = 'static/img/features'
    # Get list of all plot filenames in the directory
    plot_filenames = sorted([filename for filename in os.listdir(plots_directory) if filename.endswith('.png')])

    # Render HTML template and pass list of plot filenames
    
    
    # Show the plot
#    plt.show()
    return render_template('benignddos.html', table=html_table, plot_filenames=plot_filenames)

@app.route('/legimal')
def legimal():
    # Call load_data() to get the DataFrame
    df = load_data2()
    df.columns = df.columns.str.strip()
    clean_df=df.dropna() 
     # Map labels to numerical values (assuming 'Label' column contains categorical labels)
    #clean_df.loc[:, 'Label'] = clean_df['Label'].map({'BENIGN': 0, 'DDoS': 1})
    # data_f['Label'] = data_f['Label'].map({'BENIGN': 0, 'DDoS': 1})
    data_f1=clean_df.describe() 
    # Get the first 15 rows of the DataFrame
    # df_first_15 = data_f1.head(15)

    # Convert the first 15 rows DataFrame to HTML table
    html_table = data_f1.to_html(classes='data', header="true", index=False)
    plots_directory = 'static/img/mal/features'
    # Get list of all plot filenames in the directory
    plot_filenames = sorted([filename for filename in os.listdir(plots_directory) if filename.endswith('.png')])

    # Render HTML template and pass list of plot filenames
    
    
    # Show the plot
#    plt.show()
    return render_template('legimal.html', table=html_table, plot_filenames=plot_filenames)


@app.route('/comparision')
def comparision():
    plot_filename='static/img/modelcompare.png'
    return render_template('compare.html',plot_filename=plot_filename)

@app.route('/malcomparision')
def malcomparision():
    plot_filename='static/img/mal/modelcompare.png'
    return render_template('malcompare.html',plot_filename=plot_filename)


@app.route('/evaluation')
def evaluation():
    df = load_data()
    df.columns = df.columns.str.strip()
    data_f=df.dropna() 
    pd.set_option('use_inf_as_na', True)  # Treat inf as NaN
    null_values=data_f.isnull().sum()  # Check for NaN values
    data_f['Label'] = data_f['Label'].map({'BENIGN': 0, 'DDoS': 1})

    # Split data into features and target variable
    X = data_f.drop('Label', axis=1)
    y = data_f['Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    # Example: Check unique values of label column
    print(y_train.unique())
    print(y_test.unique())
    
    rf_accuracy,rf_f1,rf_precision,rf_recall=random_forest(X_train, X_test, y_train, y_test)
    lr_accuracy,lr_f1,lr_precision,lr_recall=logistic_regression(X_train, X_test, y_train, y_test)
    nn_accuracy,nn_f1,nn_precision,nn_recall=neural_network(X_train, X_test, y_train, y_test)
    return render_template('evaluation.html', x_train=X_train.shape,y_train=X_test.shape,rf_accuracy=f"{rf_accuracy:.4f}",rf_f1=f"{rf_f1:.4f}",rf_precision=f"{rf_precision:.4f}",rf_recall=f"{rf_recall:.4f}",lr_accuracy=f"{lr_accuracy:.4f}",lr_f1=f"{lr_f1:.4f}",lr_precision=f"{lr_precision:.4f}",lr_recall=f"{lr_recall:.4f}",nn_accuracy=f"{nn_accuracy:.4f}",nn_f1=f"{nn_f1:.4f}",nn_precision=f"{nn_precision:.4f}",nn_recall=f"{nn_recall:.4f}",random_img='static/img/random.png',logistic_img='static/img/logistic.png',neural_img='static/img/neural.png')

@app.route('/malevaluation')
def malevaluation():
    df = load_data2()
    df.columns = df.columns.str.strip()
    data_f=df.dropna() 
    pd.set_option('use_inf_as_na', True)  # Treat inf as NaN
    null_values=data_f.isnull().sum()  # Check for NaN values
    

    # Split data into features and target variable
    y=data_f['legitimate']
    data_f=data_f.drop(['legitimate'],axis=1)
    data_f=data_f.drop(['Name'],axis=1)
    data_f=data_f.drop(['md5'],axis=1)
    X = data_f

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    # Example: Check unique values of label column
    print(y_train.unique())
    print(y_test.unique())
    
    rf_accuracy,rf_f1,rf_precision,rf_recall=random_forest(X_train, X_test, y_train, y_test)
    lr_accuracy,lr_f1,lr_precision,lr_recall=logistic_regression(X_train, X_test, y_train, y_test)
    nn_accuracy,nn_f1,nn_precision,nn_recall=neural_network(X_train, X_test, y_train, y_test)
    return render_template('malevaluation.html', x_train=X_train.shape,y_train=X_test.shape,rf_accuracy=f"{rf_accuracy:.4f}",rf_f1=f"{rf_f1:.4f}",rf_precision=f"{rf_precision:.4f}",rf_recall=f"{rf_recall:.4f}",lr_accuracy=f"{lr_accuracy:.4f}",lr_f1=f"{lr_f1:.4f}",lr_precision=f"{lr_precision:.4f}",lr_recall=f"{lr_recall:.4f}",nn_accuracy=f"{nn_accuracy:.4f}",nn_f1=f"{nn_f1:.4f}",nn_precision=f"{nn_precision:.4f}",nn_recall=f"{nn_recall:.4f}",random_img='static/img/mal/random.png',logistic_img='static/img/mal/logistic.png',neural_img='static/img/mal/neural.png')

def random_forest(X_train, X_test, y_train, y_test):
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    # Getting feature importances from the trained model

    importances = rf_model.feature_importances_

    # Getting the indices of features sorted by importance
    indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=False)
    feature_names = [f"Features {i}" for i in indices]  # Replace with your column names

    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    return rf_accuracy,rf_f1,rf_precision,rf_recall

def logistic_regression(X_train, X_test, y_train, y_test):
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_precision = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    return lr_accuracy,lr_f1,lr_precision,lr_recall

def neural_network(X_train, X_test, y_train, y_test):
    nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42)
    nn_model.fit(X_train, y_train)
    nn_pred = nn_model.predict(X_test)
    nn_accuracy = accuracy_score(y_test, nn_pred)
    nn_f1 = f1_score(y_test, nn_pred)
    nn_precision = precision_score(y_test, nn_pred)
    nn_recall = recall_score(y_test, nn_pred)
    return nn_accuracy,nn_f1,nn_precision,nn_recall



if __name__ == '__main__':
    app.run(debug=True)