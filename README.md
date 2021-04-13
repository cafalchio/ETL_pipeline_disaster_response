# Disaster Response Pipeline Project
This project 

### Motivation:
This project was made as part of the Udacity course Data Science. The main motivation is learn ETL pipelines and NLP. 
The Flask app deploy the model on a local server.

### Installation:
 - pandas
 - sklearn	
 - joblib
 - sqlalchemy
 - nltk
 - plotly
 - json
 - flask

### File description:

File _ data/process_data.py
 - load data from dataframes, clean data and save into SQLite database

File _models/train_classifier.py
 - Load data from SQLite database, split data into train and validation
 - Builds a text processing and machine learning pipeline
 - Trains a Random Forest model and save the model
 
 File _app/run.py
 - Runs a Flask app that deploys the model

### Summary:


### Acknowledgements:

Audacity Data Science Course

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/://stackedit.io/).
