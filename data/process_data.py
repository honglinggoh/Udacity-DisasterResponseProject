# import libraries
import sys
import re
import numpy as np
import pandas as pd
import sqlite3
import sqlalchemy as sal
from sqlalchemy import create_engine


def load_data_file (message_filepath, categories_filepath):
    '''
    INPUT 
        message_filepath - Filepath used to load the message data for analysis
        categories_filepath - Filepath used to load the categories data for analysis
    OUTPUT
        Returns the following variables:
        df_merged - Returns the merged dataset
        X - Returns the input features.  Specifically, this is returning the messages column from the dataset
        Y - Returns the categories of the dataset.  This will be used for classification based off of the input X
        y.keys - Just returning the columns of the Y columns
    '''
    #load messages dataset    
    df_messages = pd.read_csv(message_filepath)
    
    #load categories dataset
    df_categories = pd.read_csv(categories_filepath)
    
    #Merge the messages and categories datasets using the common id
    df_merged = df_messages.merge(df_categories, how='outer', on=['id'])
    
    return df_merged


def clean_data(df):
    '''
    INPUT 
        df: Dataframe that needs data cleansing
    OUTPUT
        df: Returns a cleansed dataframe
    '''
    
    df_categories = pd.DataFrame(df['categories'])
    
    # create a dataframe of the 36 individual category columns
    df_categories_split = df_categories.categories.str.split(';', expand=True).add_prefix('cat_')

    # select the first row of the categories dataframe
    row = df_categories_split.iloc[1]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    df_categories_split.columns = category_colnames
    
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    for column in df_categories_split:
        # set each value to be the last character of the string
        df_categories_split[column] = df_categories_split[column].str[-1:]
        
        # convert column from string to numeric
        df_categories_split[column] = pd.to_numeric(df_categories_split[column])
    
    # assign the 'id' to the data frame so that we can use to merge the 2 datasets later
    df_categories_split['id'] = df['id']
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(df_categories_split, how='outer', on=['id'])
    
    # check if there are duplicate rows
    if (df.duplicated().sum()) > 0:
        # drop duplicates
        df.drop_duplicates(inplace=True) 
        
    return df


def save_data_db(df, sqldb_filepath, table_name):
    '''
    INPUT 
        df: Dataframe to be saved
        sqldb_filepath - Filepath for the database
        table_name: name of the database table for data insertion
    OUTPUT
        Saves the database
    '''
    engine = create_engine('sqlite:///{}'.format(sqldb_filepath)) 
    
    with engine.connect() as connection:
        sql_statement = "DROP TABLE IF EXISTS {}".format(table_name)
        results = connection.execute(sql_statement)
                
    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, sqldb_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data_file(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(sqldb_filepath))
        save_data_db(df, sqldb_filepath, 'MessageResponse')
        
        print('Data saved to database!')
        
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()