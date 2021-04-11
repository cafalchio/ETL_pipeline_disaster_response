import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import  RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download(['wordnet', 'punkt', 'stopwords'])

def tokenize(text):
    """
    Process text (Tokenize, Lemmatizer and normalize)
    
    Parameters:
    text sentences (str)
    
    Returns:
    List of tokens (list)
    """
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in stemmed]
    return lemmed

def load_data(database_filepath):
    '''
    Load database from sql table

    Parameters: 
        database_filepath (str): path of the database file
    Returns:
        X: messages features
        y: classification targets
        categories: labels of classification
    '''
    try:
        engine = create_engine(f'sqlite:///{database_filepath}')
        df = pd.read_sql_table('df', engine)     
        X = df.message
        y = df.iloc[:,4:].values.astype('int')
        categories = df.columns[4:]
    except:
        print(f'Data could not be loaded, check database at {database_filepath}')
        return None
    return X, y, categories

def build_model():
    """
    Returns the model
    
    Parameters:
        None
    Returns:
        sklearn pipeline model
    """
    pipeline = Pipeline([
    ('vect',TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200, max_depth = 6)))
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model using classification report 

    Parameters:
        model: model to be evaluated
        X_test: Validation X
        Y_test: Validation Y
        category_names: array of labels
    Returns:
        None
    ''' 
    Y_pred = model.predict(X_test)
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test[:, i], Y_pred[:, i],zero_division = 0))


def save_model(model, model_filepath):
    '''
    Save model in model_filepath

    Parameters: 
        model: model to be saved
        model_filepath(str): path where model will be saved

    Returns:
        None
    ''' 
    joblib.dump(model, model_filepath, compress=3)
    # pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()