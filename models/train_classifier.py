# import libraries
import sys
import re
import numpy as np
import pandas as pd

# import libraries for Database
import sqlite3
import sqlalchemy as sal
from sqlalchemy import create_engine

# import libraries for Machine Learning
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


# import libraries for model saving
import pickle
from sklearn.externals import joblib 

def load_data_db (database_filepath, table_name):
    '''
    INPUT 
        database_filepath - Filepath for the database
        table_name - Table name to load data
    OUTPUT
        X - Returns the input features - message column.
        Y - Returns the output label - set of categories columns.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df =  pd.read_sql_table(table_name, engine)
    X = df.message
    y = df.iloc[:,5:]
    return X, y

def tokenize(text):
    '''
    INPUT 
        text: raw data to process   
    OUTPUT
        Returns a processed data
    '''
    
    # replace the non char & number in the text with empty space 
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    
    # tokenize the word
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatize the words and remove space
    lemmed = [WordNetLemmatizer().lemmatize(w).strip() for w in words]
    return lemmed

def build_model(X_train,y_train):
    '''
    INPUT 
        X_Train: Training features
        y_train: Training labels
    OUTPUT
        Returns a trained model
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
    }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    cv.fit(X_train,y_train)
    return cv

def evaluate_model(pipeline, X_test, y_test):
    '''
    INPUT 
        pipeline: Trained model pipeline
        X_test: Test features
        y_test: Test labels
    OUTPUT
        Print the f1 score, precision and recall for each output category of the dataset
    '''
    # predict on test data
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=y_test.keys()))

def save_model(model, pickle_filename):
    '''
    INPUT 
        model: The model to be saved
        pickle_filename: Pickle file name
    OUTPUT
        None
    '''  
    
    # Save the model as a pickle in a file 
    joblib.dump(model, pickle_filename)

def main():
    if len(sys.argv) == 3:
        database_filepath, pickle_filename = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data_db (database_filepath,'MessageResponse')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train,y_train)
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)
        
        print('Saving model...\n    MODEL: {}'.format(pickle_filename))
        save_model(model, pickle_filename)

        print('Trained model saved!')     
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()