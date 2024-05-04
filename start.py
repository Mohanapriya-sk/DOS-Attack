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

def dos_calc():
    df = pd.read_csv("data/DDos.csv")
    # Remove the spaces before the column names
    df.columns = df.columns.str.strip()

    #Unique values in the Label target column
    df.loc[:,'Label'].unique()

    #Checking the null values in the dataset.
    plt.figure(1,figsize=( 10,4))
    plt.hist( df.isna().sum())
    # Set the title and axis labels
    plt.xticks([0, 1], labels=['Not Null=0', 'Null=1'])
    plt.title('Columns with Null Values')
    plt.xlabel('Feature')
    plt.ylabel('The number of features')

    # Save the plot as a static image file
    plot_filename2 = 'static/img/null_plot.png'
    plt.savefig(plot_filename2)
    plt.close()
    plotMissingValues(df)
    ## Removing the null values
    data_f=df.dropna()
    #Checking the null values in the dataset.
    plt.figure(1,figsize=( 10,4))
    plt.hist(data_f.isna().sum())
    # Set the title and axis labels
    plt.xticks([0, 1], labels=['Not Null=0', 'Null=1'])
    plt.title('Data after removing the Null Values')
    plt.xlabel('The number of null values')
    plt.ylabel('Number of columns')

    # Show the plot
    # plt.show()
    # Save the plot as a static image file
    plot_filename3 = 'static/img/removenull_plot.png'
    plt.savefig(plot_filename3)
    plt.close()
    pd.set_option('use_inf_as_na', True)  # Treat inf as NaN
    null_values=data_f.isnull().sum()  # Check for NaN values
    # To know the data types of the columns

    # (data_f.dtypes=='object')
    # Convert the labels in the DataFrame to numerical values
    # data_f['Label'] = data_f['Label'].map({'BENIGN': 0, 'DDoS': 1})
     # Map labels to numerical values (assuming 'Label' column contains categorical labels)
    data_f.loc[:, 'Label'] = data_f['Label'].map({'BENIGN': 0, 'DDoS': 1})
    # Print the DataFrame
    plt.figure(1,figsize=( 10,4))
    plt.hist(data_f['Label'], bins=[0, 0.3,0.7,1], edgecolor='black')  # Specify bins as [0, 1]
    plt.xticks([0, 1], labels=['BENIGN=0', 'DDoS=1'])
    plt.title("BENIGN VS. M-DOS")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    # plt.show()
    plot_filename4 = 'static/img/benignddos_plot.png'
    plt.savefig(plot_filename4)
    plt.close()
    output_directory = 'static/img/features'  # Directory to save the plots
    create_and_save_histograms(data_f, output_directory)

    X = data_f.drop('Label', axis=1)
    y = data_f['Label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
   

        # Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    # Getting feature importances from the trained model

    importances = rf_model.feature_importances_

    # Getting the indices of features sorted by importance
    indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=False)
    feature_names = [f"Features {i}" for i in indices]  # Replace with your column names
    # Getting feature importances from the trained model
    importances = rf_model.feature_importances_

    # Getting the indices of features sorted by importance
    indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=False)
    feature_names = [f"Features {i}" for i in indices]  # Replace with your column names
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    plot_confusion_matrix(y_test, rf_pred, ['Benign', 'DDoS'], 'Random Forest Confusion Matrix','static/img/random.png')

    #Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_precision = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    plot_confusion_matrix(y_test, lr_pred, ['Benign', 'DDoS'], 'Logistic Regression Confusion Matrix','static/img/logistic.png')

    #Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42)
    nn_model.fit(X_train, y_train)
    nn_pred = nn_model.predict(X_test)
    nn_accuracy = accuracy_score(y_test, nn_pred)
    nn_f1 = f1_score(y_test, nn_pred)
    nn_precision = precision_score(y_test, nn_pred)
    nn_recall = recall_score(y_test, nn_pred)

    # Confusion Matrix for Neural Network
    plot_confusion_matrix(y_test, nn_pred, ['Benign', 'DDoS'], 'Neural Network Confusion Matrix','static/img/neural.png')

    #Model Comparision
    # Random Forest
    rf_proba = rf_model.predict_proba(X_test)

    # Logistic Regression
    lr_proba = lr_model.predict_proba(X_test)
    
    # Neural Network
    nn_proba = nn_model.predict_proba(X_test)
    # Combine predictions for ROC curve
    # Calculate ROC curve for Random Forest
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba[:, 1])
    rf_auc = auc(rf_fpr, rf_tpr)

    # Calculate ROC curve for Logistic Regression
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba[:, 1])
    lr_auc = auc(lr_fpr, lr_tpr)



    # Calculate ROC curve for Neural Network
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_proba[:, 1])
    nn_auc = auc(nn_fpr, nn_tpr)

    # Plot ROC curves for all models
    plt.figure(figsize=(8, 6))
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
    plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
    plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')

    # Plot ROC curve for random classifier (50% area)
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier (AUC = 0.50)')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()
    # plt.show()
    plot_filename8 = 'static/img/modelcompare.png'
    plt.savefig(plot_filename8)
    plt.close()





# Function to generate and display a detailed confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, title,plot_filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.show()

    plt.savefig(plot_filename)
    plt.close()

def create_and_save_histograms(dataframe, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each column (feature) in the DataFrame
    for idx, col in enumerate(dataframe.columns):
        # Create a new figure for each feature
        plt.figure(idx + 1)  # Increment figure number

        # Plot histogram for the current feature
        plt.hist(dataframe[col])
        plt.title(col)  # Set title as feature name

        # Save the plot with a filename based on the feature name
        filename = os.path.join(output_dir, f'figure_{idx + 1}.png')
        plt.savefig(filename)
        plt.close()  # Close the plot to release resources

    # print(f'{len(dataframe.columns)} histogram plots saved in {output_dir}')

def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()  # Counting null values for each column
    fig = plt.figure(figsize=(16, 5))
    missing_values.plot(kind='bar')
    plt.xlabel("Features")
    plt.ylabel("Missing values")
    plt.title("Total number of Missing values in each feature")
    # plt.show()
    # Save the plot as a static image file
    plot_filename = 'static/img/miss_plot.png'
    plt.savefig(plot_filename)
    plt.close()
dos_calc()
print("Processing completed..")