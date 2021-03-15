import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    ''' 
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    df = messages.merge(categories)
    categories = pd.DataFrame(list(df.categories.apply(lambda x: x.split(';'))))
    categories.columns = [n.split('-')[0] for n in categories.iloc[0:1].values[0]]
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
    categories = categories.apply(pd.to_numeric)
    df.drop('categories' ,inplace=True, axis =1)
    df = pd.concat( (df, categories), axis = 1)
    return df

def clean_data(df):
    ''' '''
    
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    try:
        engine = create_engine(f'sqlite:///{database_filename}.db')
        df.to_sql('df', engine, index=False)
        print(f'Saved in {database_filename} database.')
        return True
    except:
        print('ERROR, data could not be saved!')
        return False


def main():
    if len(sys.argv) == 4:

        (messages_filepath, categories_filepath, database_filepath) = sys.argv[1:]
        print(messages_filepath)
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()