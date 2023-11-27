# MLProject1
Repo for CSI 5810 Information Retrieval and Knowledge Discovery project 1 at Oakland University
# Use
Before running the code, make sure to install the necessary dependencies\
Note that the instructions below are for linux systems, if you want to read more about virtual environments for other systems, please click [here](https://docs.python.org/3/library/venv.html)\
To initialize a new virtual environment, run the following command:
```
python3 -m venv .venv
```
To activate the virtual environment, run the following command:
```
. .venv/bin/activate
```
To deactivate the virtual environment, run the following command:
```
deactivate
```
To install all dependencis, run the following command:
```
pip install -r requirements.txt
```
Once the requirements are installed, run the following command:
```
python3 final.py
```
Note that the default behavior will first display 3 windows for data visualization.\
Closing these windows will open 3 more for KNN metrics.\
Closing these windows will open 3 more for Random Forest Classifier Metrics.\
Comment\uncomment the last three lines in ``final.py`` to run specific parts of the project.
