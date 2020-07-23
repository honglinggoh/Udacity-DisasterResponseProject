# Disaster Response Pipeline Project:
 This project is to build a multi classifier model for an API that classifies disaster messages via a Web Page.

## Project Files: 
1. app folder
	- run.py - read the data from database and prepare graph configuration to display in the web page
    - templates folder - HTML files for the display
2. data folder
	- disaster_categories.csv & disaster_messages.csv - raw data files for the disaster messages & response used to train the model
	- DisasterResponse.db - sqllite database to store the process data
    - process_data.py - perform the ETL pipeline to process the data from raw data files and stored into database
3. models folder
	- train_classifier.py - read the data from database and perform Machine Learning pipeline to build the model
    - classifier.pkl - stored the trained model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
