from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
from final import predict_rfc

app = Flask(__name__)

def train_random_forest():
    data = pd.read_csv('./train.csv', index_col=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    n_estimators = 60

    n_set = range(60, 65, 10)
    accuracy_values = []

    kf = KFold(n_splits=10)

    accuracy = []
    precisions = []
    recalls = []

    for train_index, test_index in kf.split(X):
        xtrain, xtest = X.iloc[train_index], X.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

        rf = RandomForestClassifier(n_estimators=60, random_state=0)

        rf.fit(xtrain, ytrain)

        accuracy.append(accuracy_score(ytest, rf.predict(xtest)))
        precisions.append(precision_score(ytest, rf.predict(xtest), average='weighted'))
        recalls.append(recall_score(ytest, rf.predict(xtest), average='weighted'))

    return accuracy, precisions, recalls


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points']

        wilderness_area_features = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']
        soil_type_features = [f'Soil_Type{i}' for i in range(1, 41)]

        user_input = np.array([request.form[feature] for feature in features])
        
        bin_cols = [0 for _ in range(0,44)]

        user_input = np.concatenate([user_input, np.zeros(44)])

        print(f'soil type field: {request.form["Soil_Type"]}')
        print(f'type: {type(request.form["Soil_Type"])}')
        user_input[int(request.form["Soil_Type"])] = 1
        user_input[int(request.form["Wilderness_Area"])] = 1

        user_input = user_input.reshape(1,-1)
        prediction_num = int(predict_rfc(user_input))
        classes = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
        prediction = classes[prediction_num]
        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
