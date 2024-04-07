# SMS Spam Classifier
End to end code for the sms spam classifier project. In this project, we will predict whether a given message is a spam message or not. This can be used for classifying normal messages from spam messages, which are generally sent by companies to their customers. However, our device automatically detects the SMS and puts it into the spam folder.


## How does this project work?
First download the file using this [link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download) and put it in the same directory where the app.py and main.py then create a virtual env outside of this directory using this command `python -m venv <your-venv-name>` after that run the command `pip install -r requirements.txt` run main.py after that you can use the command to runserver of streamlit.

## Steps for the project

### 1. Data Cleaning
_____

Remove all the unnecessary values from the data like Nan etc.
In `spam.csv` if we put `df.info()` we will see that we have 5 rows and three rows have a lot of null values which is not for analysis, so we remove these three rows(which are under a in a column table) rows.
`df.drop(columns=['Unnamed : 2', 'Unnamed : 3', 'Unnamed : 4'], inplace = True)`
after this we get only have 2 columns now v1, v2 as these names are not descriptive I will change them `df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)` now in our data the sms which is not spam is saved as 'ham' and spam save as 'spam' convert them into number like if it is not spam then 0 if it spam then 1 for that I will use LabelEncoder
`from sklearn.preprocessing import LabelEncoder` then make a instance of it `encoder = LableEncoder()` then apply it on 'target 1' column. `encoder.fit_transform(df['target'])` now for implementing this into our data use this command `df['target'] = encoder.fit_transform(df['target])` and now if you check data using `df.head()` the target column now contain 0's and 1's now. 
After this always check these two things during data cleaning..

<b> Check Missing Values </b> : 
Using this command `df.isnull().sum()` you can check weather df have any missing value or not.
<br /><br />
<b> Check Duplicate Values </b> :
Using this command `df.duplicated().sum()` we can get the sum of duplicate values in our dataframe. If you find any duplicate values then remove it using `df = df.drop_duplicates(keep='first')`

### 2. EDA (Exploratory Data Analysis )
____

It is a process used to analyze and summarize datasets, identifying characteristics, patterns, and relationships in the data before applying machine learning techniques.

Now we will check how many sms are spam and how many are not spam. for that run command `df['target'].values_counts()` or you can also see it in form of diagram `import matplotlib.pyplot as plt`; `plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")`; `plt.show()` When we apply it on this df we see that 88% of data is not spam and remaining are spam. That's lead to data imbalance condition. <br />
Now we will check our sms that how many characters,words,sentences in the sms.
For that we will create 3 column. 1. characters, 2. words, 3. sentences

We will use nltk like this : `import nltk`; `nltk.download('punkt')`
Calculate characters and create new column for it : `df['num_characters'] = df['text'].apply(len)`
Calculate words and create new column for it : `df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))`

Calculate sentences and create new column for it : `df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))`

Also, you can get all the details of spam and non-spam messages.

- For spam messages : `df[df['target']==1][['num_characters', 'num_words', 'num_sentences']].describe()`

- For non-spam messages : `df[df['target']==0][['num_characters', 'num_words', 'num_sentences']].describe()`

### 3. Data Preprocessing
___

- Lower case
- Tokenization: Break sentence into words.
- Removing special characters:  It's because they don't impact on data. 
- Removing stop words and punctuation:  It's because they don't impact on data. 
- Stemming:  only root word will be there

For this we will make a function:

````
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import string
    
    def data_preprocessing(text):
        # lower case
        text =  text.lower()
        
        # tokenization
        text = nltk.word_tokenize(text)
        
        # removing special characters
        processed_words = []
        for i in text:
            if i.isalnum():
                processed_words.append(i)
        text = processed_words[:] # you can't copy like text = y bcz list is a mutable data type
        processed_words.clear()
        
        # removing stop words amd punctuation 
        ps = PorterStemmer()
        for word in text:
            if word not in stopwords.words('english') and word not in string.punctuation:
                processed_words.append(word)
        
        text = processed_words[:]
        processed_words.clear()
        
        # stemming
        for i in text:
            processed_words.append(ps.stem(i))
        
        return " ".join(processed_words)
````

Apply this function on text and make a new column:
    `df['transformed_text'] = df['text'].apply(transform_text)`


Top 30 words in spam sms and non spam sms:

```
    from collections import Counter
    spam_corpus = []
    for msg in df[df['target'] == 1]['transformed_text'].tolist():
        for word in msg.split():
            spam_corpus.append(word)
            
    pd.Dataframe(Counter(spam_corpus).most_common(30)) # this will give us the key value pairs (word : <therir occurence>)
    
    # can make barplot
    sns.barplot(pd.Dataframe(Counter(spam_corpus).most_common(30)[0], pd.DataFrame(Counter(spam_corpus).most_common(30)[1])
    plt.xtics(rotation='vertical')
    plt.show()
    
    non_spam_corpus = []
    for msg in df[df['target'] == 0]['transformed_text'].tolist():
        for word in msg.split():
            non_spam_corpus.append(word)
            
    pd.Dataframe(Counter(non_spam_corpus).most_common(30)) # this will give us the key value pairs (word : <therir occurence>)
    sns.barplot(pd.Dataframe(Counter(non_spam_corpus).most_common(30)[0], pd.DataFrame(Counter(non_spam_corpus).most_common(30)[1])
    plt.xtics(rotation='vertical')
    plt.show()
```

### 4. Model building
For Solving a problem we build a machine learning model

<b>Firstly we make a model based on naive based algorithm because it's good for text data </b>

Firstly we start with vectorization like this:
```
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(df["transformed_text"]).toarray()
    Y = df["target"].values

    # Now train these models

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()

# test for GaussianNavieBaseAlgo
_________________________________________

    gnb.fit(X_train, y_train)
    y_pred1 = gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred1)) # Output: 0.87814..
    print(confusion_matrix(y_test, y_pred1)) # Output: [[790 106][20 118]]
    print(precision_score(y_test, y_pred1)) # Output: 0.5267...

# GNB RESULTS : It's performing very poorly
==============================================================

# test for MultinomialNavieBaseAlgo
_______________________________________________________

    mnb.fit(X_train, y_train)
    y_pred2 = mnb.predict(X_test)
    print(accuracy_score(y_test, y_pred2)) # Output: 0.965...
    print(confusion_matrix(y_test, y_pred2)) # Output: [[872 24][12 126]]
    print(precision_score(y_test, y_pred2)) # Output: 0.84
    
    # MNB RESULTS : It's accuracy is good but precision score is still low.  Bcz it is imbalance data so precision will take more priority.

=========================================================================
# test for BernoulliNavieBaseAlgo
_______________________________________________________

    bnb.fit(X_train, y_train)
    y_pred3 = mnb.predict(X_test)
    print(accuracy_score(y_test, y_pred3)) # Output: 0.965...
    print(confusion_matrix(y_test, y_pred3)) # Output: [[872 24][12 126]]
    print(precision_score(y_test, y_pred3)) # Output: 0.84
    
    # BNB RESULTS : It's accuracy is good and precision score is also good.
    
    # Now if instead of CountVectorizer we use TfidfVectorizer like this :
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer()
        
        X = tfidf.fit_transform(df["transformed_text"]).toarray()
        Y = df["target"].values
        ....remaining code will be same.....
        
        >>> Coming from improvements section <<<<
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=3000)
        
        # appending the num_character col to X
        X = tfidf.fit_transform(df["transformed_text"]).toarray()
        Y = df["target"].values
        
        X = np.hstack((X, df['num_characters'].values.reshape(-1,1)))
        ....remaining code will be same.....
        
        
     then MNB is giving the precison of 1.0 so we are selecting MNB. 
```

### 5. Evaluation
Evaluate created model 

Now I will compare this MNB with some more models. In this I will train the ML models and record the results one by one. So side by side comparison will be possible. Most models are classification algorithms

```
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier
    
    # object of all algorithms
    svc = SVC(kernel='sigmoid', gamma=1.0)
    knc = KNeighborsClassifier()
    mnb = MultinomialNB()
    dtc = DecisionTreeClassifier(max_depth=5)
    lrc = LogisticRegression(solver='liblinear', penalty='l1')
    rfc = RandomForestClassifier(n_estimators=50, random_state=2)
    abc = AdaBoostClassifier(n_estimators=50, random_state=2)
    bc = BaggingClassifier(n_estimators=50, random_state=2)
    etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
    gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
    xgb = XGBClassifier(n_estimators=50,random_state=2)
    
    
    # Created a dictionary where keys are the name of algorithm and in values the object of that algorithm.
    clfs = {
        'SVC' : svc,
        'KN' : knc, 
        'NB': mnb, 
        'DT': dtc, 
        'LR': lrc, 
        'RF': rfc, 
        'AdaBoost': abc, 
        'BgC': bc, 
        'ETC': etc,
        'GBDT':gbdt,
        'xgb':xgb
    }
    
    # Create a function which willl take three parameters: classifier, trained X data, trained Y data
    
    def train_classifier(clf,X_train,y_train,X_test,y_test):
        clf.fit(X_train,y_train) # train classifer on the provided X, Y data
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        
        return accuracy,precision
        
    # Example for running this function    
    # train_classifier(svc,X_train,y_train,X_test,y_test)
    
    # Loop through all the algorithms and take one algorithm at a time and train that model on the data and store the accuracy and precision score for every algorithm.
    
    accuracy_scores = []
    precision_scores = []
    
    for name,clf in clfs.items():
        current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
        print("For ",name)
        print("Accuracy - ",current_accuracy)
        print("Precision - ",current_precision)
        accuracy_scores.append(current_accuracy)
        precision_scores.append(current_precision)
    
    # Convert this into a dataframe and also sort by most accrate algorithm
    performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
    
    performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
```

you can plot a diagram for see the accuracy and precision for each algorithm like this:

```
    sns.catplot(x = 'Algorithm', y='value', 
                   hue = 'variable',data=performance_df1, kind='bar',height=5)
    plt.ylim(0.5,1.0)
    plt.xticks(rotation='vertical')
    plt.show()
```

### 6. Improvements
Improvement on our model. In tfidf vectorizer give us a feature that for how many words we want to vectorize. Before we are giving all the words but we can restrict the words limit using 'max'

Now In end of Model Building I change this:

```
    from sklearn.feature_extraction.text import TfidfVectorizer 
    tfidf = TfidfVectorizer()
```

to this:

```
   from sklearn.feature_extraction.text import TfidfVectorizer 
   tfidf = TfidfVectorizer(max_features=3000) 
```

Now with precision_scores 3000
```
    temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)
    
    temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)
    
    # We have before, after accuracy and precion in new columns for 'TfidfVectorizer(max_features=3000)'
     
    new_df = performance_df.merge(temp_df,on='Algorithm')
    # After this the most powerful algoritm is Naive Base algoritm  
    new_df_scaled = new_df.merge(temp_df,on='Algorithm')
    temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)
    new_df_scaled.merge(temp_df,on='Algorithm')
    
    # Create combination of best performing model then let's see this combination can outperform NB or not
    # Voting Classifier
    svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
    mnb = MultinomialNB()
    etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
    
    from sklearn.ensemble import VotingClassifier
    
    voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
    
    voting.fit(X_train,y_train)
    
    VotingClassifier(estimators=[('svm',
                                  SVC(gamma=1.0, kernel='sigmoid',
                                      probability=True)),
                                 ('nb', MultinomialNB()),
                                 ('et',
                                  ExtraTreesClassifier(n_estimators=50,
                                                       random_state=2))],
                     voting='soft')
    
    y_pred = voting.predict(X_test)
    print("Accuracy",accuracy_score(y_test,y_pred))
    print("Precision",precision_score(y_test,y_pred)) # low precison 0.99..
    
    # Applying stacking
    estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
    final_estimator=RandomForestClassifier()
    
    from sklearn.ensemble import StackingClassifier
    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy",accuracy_score(y_test,y_pred))
    print("Precision",precision_score(y_test,y_pred))

```
Now we tried many classifier algoritms performs various operation and till now MNB is performing the best giving 1.0 prevision. 

Extract the trained model
```
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))
```
