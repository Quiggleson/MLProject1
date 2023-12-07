import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

def cleaning():
    
    data = pd.read_csv('./train.csv',index_col=0)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    
    fig, axs = plt.subplots(4,3,layout='constrained')

    print('Checking NaN:')
    for col in data:
        res = data[col].isnull().values.any()
        if res:
            print(f'NaN values exist in {col}')
    
    print('Checking Zeroes:')
    for col in data:
        res = np.any(data[col] == 0)
        if res:
            print(f'Zero exist in {col}')
    
    print('Checking binary cols')
    
    wilderness_cols = X.iloc[:,10:14].to_numpy()
    soil_cols = X.iloc[:,14:54].to_numpy()
    
    if not np.any(wilderness_cols.any(axis=1)):
        print('No data for some wilderness examples')
    
    if not np.any(soil_cols.any(axis=1)):
        print('No data for some soil examples')
    
    # Duplicate entries
    unique_rows = np.unique(data, axis=0)
    if data.shape == unique_rows.shape:
        print('No duplicates exist')
    else:
        print('Duplicate entries exist!')
    
    # Histograms
    xlabels = ["Elevation in meters", "Aspect in degrees azimuth", "Slope in degrees", "Horizontal Distance to nearest surface water features", "Vertical Distance to nearest surface water features", "Horizontal Distance to nearest roadways", "Hillshade index at 9am, summer solstice", "Hillshade index at noon, summer solstice", "Hillshade index at 3pm, summer solstice", "Horizontal Distance to nearest wildfire ignition points"]
    
    # Histograms for first 10 attributes
    for i, (col, xlabel) in enumerate(zip(data.iloc[:,:10], xlabels)):
        axs[i % 4, i // 4].hist(data[col], bins=15)
        axs[i % 4, i // 4].set_xlabel(xlabel)
        axs[i % 4, i // 4].set_ylabel("Number of examples")
        axs[i % 4, i // 4].set(title=f'Histogram of {col}')
    
    # Bar graph for wilderness area
    category_counts = np.sum(wilderness_cols, axis=0)
    
    category_names = [f'Wilderness_Area{i+1}' for i in range(wilderness_cols.shape[1])]
    
    # Plotting the bar graph
    axs[2,2].bar(category_names, category_counts, color='red')
    axs[2,2].set_xlabel('Wilderness Areas')
    axs[2,2].set_ylabel('Number of Examples')
    axs[2,2].set(title='Number of Examples in Each Wilderness Area')
    axs[2,2].set_xticklabels(category_names, rotation=45)
    
    # Bar graph for soil type
    category_counts = np.sum(soil_cols, axis=0)
    
    category_names = [f'{i+1}' for i in range(soil_cols.shape[1])]
    
    # Plotting the bar graph
    axs[3,2].bar(category_names, category_counts, color='red')
    axs[3,2].set_xlabel('Soil Types')
    axs[3,2].set_ylabel('Number of Examples')
    axs[3,2].set(title='Number of Examples with Each Soil Type')
    axs[3,2].set_xticklabels(category_names, rotation=45)
    
    labels = data.iloc[:, -1]
    
    # Count occurrences of each label
    label_counts = labels.value_counts().sort_index()
    
    # Plotting the bar graph for the labels distribution
    plt.figure()
    plt.bar(['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'], label_counts.values)
    plt.xticks(rotation=45)
    plt.xlabel('Forest Cover Type')
    plt.ylabel('Number of Examples')
    plt.title('Number of Examples in Each Cover Type')
    plt.gcf().subplots_adjust(bottom=0.3)
    
    correlation_matrix = data.iloc[:,:10].corr()
    
    # Plotting a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)
    plt.show()

def knn():    
    data = pd.read_csv('./train.csv',index_col=0)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    
    kf = KFold(n_splits=10)
    
    results = []
    recalls = []
    precisions = []
    
    for train_index, test_index in kf.split(X):
        xtrain, xtest = X.iloc[train_index], X.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
    
        knn = KNeighborsClassifier(1)
    
        knn.fit(xtrain, ytrain)
    
        ypred = knn.predict(xtest)
    
        results.append(accuracy_score(ytest, ypred))
        recalls.append(recall_score(ytest, ypred, average='weighted'))
        precisions.append(precision_score(ytest, ypred, average='weighted'))
    
    plt.bar(range(1, 11), results)
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy with Different Folds')
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.text(0,-0.2,f'Mean: {np.mean(results)}\nStandard Deviation:{np.std(results)}')
    
    plt.figure()
    plt.bar(range(1, 11), precisions)
    plt.xlabel('Fold')
    plt.ylabel('Precision')
    plt.title('Precision with Different Folds')
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.text(0,-0.2,f'Mean: {np.mean(precisions)}\nStandard Deviation:{np.std(precisions)}')
    
    plt.figure()
    plt.bar(range(1, 11), recalls)
    plt.xlabel('Fold')
    plt.ylabel('Recall')
    plt.title('Recall with Different Folds')
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.text(0,-0.2,f'Mean: {np.mean(recalls)}\nStandard Deviation:{np.std(recalls)}')
    plt.show()

def rfc(plot = False):

    data = pd.read_csv('./train.csv',index_col=0)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    
    n_estimators = 60
    
    n_set = range(60,65,10)
    accuracy_values = []
    
    kf = KFold(n_splits=10)
    
    accuracy = []
    precisions = []
    recalls = []
    
    for train_index, test_index in kf.split(X):
        xtrain, xtest = X.iloc[train_index], X.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
    
        rf = RandomForestClassifier(n_estimators=60,random_state=0)
    
        rf.fit(xtrain,ytrain)
    
        accuracy.append(accuracy_score(ytest, rf.predict(xtest)))
        precisions.append(precision_score(ytest, rf.predict(xtest), average='weighted'))
        recalls.append(recall_score(ytest, rf.predict(xtest), average='weighted'))
    
    dump(rf, 'trainedrf.joblib')
    
    if plot:
        plt.bar(range(1, 11), accuracy)
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy with Different Folds')
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.text(0,-0.2,f'Mean: {np.mean(accuracy)}\nStandard Deviation:{np.std(accuracy)}')
        
        plt.figure()
        plt.bar(range(1, 11), precisions)
        plt.xlabel('Fold')
        plt.ylabel('Precision')
        plt.title('Precision with Different Folds')
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.text(0,-0.2,f'Mean: {np.mean(precisions)}\nStandard Deviation:{np.std(accuracy)}')
        
        plt.figure()
        plt.bar(range(1, 11), recalls)
        plt.xlabel('Fold')
        plt.ylabel('Recall')
        plt.title('Recall with Different Folds')
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.text(0,-0.2,f'Mean: {np.mean(recalls)}\nStandard Deviation:{np.std(recalls)}')
        
        plt.show()

def predict_rfc(df):
    rf = load('trainedrf.joblib')
    prediction = rf.predict(df)
    return prediction


# Uncomment the part of the project you'd like to see

# Data visualization and checking for bad data
# cleaning()

# KNN Classifier
# knn()

# Random Forest Classifier
rfc()
