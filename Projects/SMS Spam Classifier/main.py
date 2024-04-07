import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tfidf = TfidfVectorizer
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)
ps = PorterStemmer()
encoder = LabelEncoder()
cv = CountVectorizer()


def data_preprocessing(text):
    # lower case
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # removing special characters
    processed_words = []
    for i in text:
        if i.isalnum():
            processed_words.append(i)
    text = processed_words[:]  # you can't copy like text = y bcz list is a mutable data type
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


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')  # encoding added to read the csv file
# print(df.sample(5))
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
# df.sample(5)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
# df.sample(5)

df['target'] = encoder.fit_transform(df['target'])
df = df.drop_duplicates(keep='first')

nltk.download('punkt')
nltk.download('stopwords')

df['num_characters'] = df['text'].apply(len)

# num of words
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# df[['num_characters','num_words','num_sentences']].describe()
# df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()
# df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()

df['transformed_text'] = df['text'].apply(data_preprocessing)
spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

non_spam_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        non_spam_corpus.append(word)

# print(df.head)

X = tfidf().fit_transform(df['transformed_text']).toarray()
# print(x.shape)

Y = df["target"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, Y_train)
y_pred1 = gnb.predict(X_test)

mnb.fit(X_train, Y_train)
y_pred2 = mnb.predict(X_test)

bnb.fit(X_train, Y_train)
y_pred3 = bnb.predict(X_test)

clfs = {
    'SVC': svc,
    'KN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}


def train_classifier(clf, X_train, Y_train, X_test, Y_test):
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)

    return accuracy, precision


train_classifier(svc, X_train, Y_train, X_test, Y_test)

accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, Y_train, X_test, Y_test)

    print("For ", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame(
    {'Algorithm': clfs.keys(), 'Accuracy': accuracy_scores, 'Precision': precision_scores}).sort_values('Precision',
                                                                                                        ascending=False)

performance_df1 = pd.melt(performance_df, id_vars="Algorithm")

# sns.catplot(x = 'Algorithm', y='value',
#                hue = 'variable',data=performance_df1, kind='bar',height=5)
# plt.ylim(0.5,1.0)
# plt.xticks(rotation='vertical')
# plt.show()

temp_df = pd.DataFrame({'Algorithm': clfs.keys(), 'Accuracy_max_ft_3000': accuracy_scores,
                        'Precision_max_ft_3000': precision_scores}).sort_values('Precision_max_ft_3000',
                                                                                ascending=False)

temp_df = pd.DataFrame(
    {'Algorithm': clfs.keys(), 'Accuracy_scaling': accuracy_scores, 'Precision_scaling': precision_scores}).sort_values(
    'Precision_scaling', ascending=False)

new_df = performance_df.merge(temp_df, on='Algorithm')

new_df_scaled = new_df.merge(temp_df, on='Algorithm')

temp_df = pd.DataFrame({'Algorithm': clfs.keys(), 'Accuracy_num_chars': accuracy_scores,
                        'Precision_num_chars': precision_scores}).sort_values('Precision_num_chars', ascending=False)

new_df_scaled.merge(temp_df, on='Algorithm')

# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)], voting='soft')
voting.fit(X_train, Y_train)

VotingClassifier(estimators=[('svm',
                              SVC(gamma=1.0, kernel='sigmoid',
                                  probability=True)),
                             ('nb', MultinomialNB()),
                             ('et',
                              ExtraTreesClassifier(n_estimators=50,
                                                   random_state=2))],
                 voting='soft')

y_pred = voting.predict(X_test)
print("Accuracy", accuracy_score(Y_test, y_pred))
print("Precision", precision_score(Y_test, y_pred))

estimators = [('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator = RandomForestClassifier()

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
print("Accuracy", accuracy_score(Y_test, y_pred))
print("Precision", precision_score(Y_test, y_pred))

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
