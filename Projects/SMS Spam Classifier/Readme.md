# SMS Spam Classifier
In this project, we will predict whether a given message is a spam message or not. This can be used for classifying normal messages from spam messages, which are generally sent by companies to their customers. However, our device automatically detects the SMS and puts it into the spam folder.


## How does this project work?
First download the file using this [link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download) and put it in the same directory where the app.py and main.py then create a virtual env outside of this directory using this command `python -m venv <your-venv-name>` after that run the command `pip install -r requirements.txt` run main.py after that you can use the command to runserver of streamlit.

## Steps for the project

### 1. Data Cleaning
Remove all the unnecessary values from the data like Nan etc.
### 2. EDA (Exploratory Data Analysis )
It is a process used to analyze and summarize datasets, identifying characteristics, patterns, and relationships in the data before applying machine learning techniques.

### 3. Text Preprocessing
Converts strings to lists.
Extracts relevant information like genres, keywords, cast, and crew.
Cleans and preprocesses text data (e.g., removing spaces, stemming).

### 4. Vectorization:
Utilizes CountVectorizer to convert text data into numerical vectors.
Calculates cosine similarity between vectors to determine movie similarity.

### 5. Model building
Create model

### 6. Evaluation
Evaluate created model 

### 7. Improvements
Improvement on our model