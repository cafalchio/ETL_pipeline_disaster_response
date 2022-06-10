# Disaster Response Pipeline Project

The  pipeline + model can to classify emergency text into one of the 26 categories and display the result in a Flask web app. 

to do: update the model to a BERT related model (maybe Roberta)

### **Motivation:**

The major motivation was to learn and practice ETL pipelines and text processing. 
To be able to make a user interface, Flask app was created to run in a local server


### **Requirements:**

 - pandas
 - sklearn	
 - joblib
 - sqlalchemy
 - nltk
 - plotly
 - json
 - flask

### **Files description:**

`process_data.py`, a data cleaning pipeline that:

-   Loads the `messages` and `categories` datasets
-   Merges the two datasets
-   Cleans the data
-   Stores it in a SQLite database

`train_classifier.py`, a machine learning pipeline that:

-   Loads data from the SQLite database
-   Splits the dataset into training and test sets
-   Builds a text processing and machine learning pipeline
-   Trains and tunes a model using GridSearchCV
-   Outputs results on the test set
-   Exports the final model as a pickle file
 
 File _app/run.py
 - Runs a Flask app that deploys the model


### **Instructions:**

>>>>>>> Stashed changes
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/tree_class`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/://stackedit.io/).
<<<<<<< Updated upstream
=======

### **A screenshot of the Dashboard**

![print screen](https://github.com/cafalchio/ETL_pipeline_disaster_response/blob/main/screen.PNG)
