# Disaster Response Pipeline Project

The idea of this pipeline + model is to classify emergency text into one of the 26 categories and display the result in a Flask web app. 

### **Motivation:**

The major motivation was to learn and practice ETL pipelines and text processing. To complete the project, use a Flask app to deploy the model in a local server.


### **Installation:**

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

![print screen](https://github.com/cafalchio/ETL_pipeline_disaster_response/blob/main/screen.PNG)

### **Acknowledgements:**

Thanks to Audacity Data Science Course instructors and revisers.
