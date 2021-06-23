"""
@author: Salvatore Calderaro
@author: Simone Contini
"""

import pandas as pd
import configobj
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import spacy
from sklearn.metrics import accuracy_score

config = configobj.ConfigObj('env.b')

data_path = config['MODEL_TRAIN_PATH']
vec_path = config['MODEL_VEC_PATH']
model_path = config['MODEL_PATH']



"""
Funzione che prende in input il dataset per l'addestramento ed effettua il
cleaning dei dati
"""
def cleaning(path):
    data = pd.read_csv(path)

    for i,row in data.iterrows():
        row['Text'] = re.sub('\S*@\S*\s?', '', str(row['Text']))
        row['Text'] = re.sub('\s+', ' ', str(row['Text']))
        row['Text'] = re.sub("\'", "", str(row['Text']))
        row['Text'] = re.sub(r'http\S+', ' ', str(row['Text']))
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002500-\U00002BEF"
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
        row['Text'] = emoji_pattern.sub(r'', row['Text'])
        data.loc[i, 'Text'] = row['Text']

    return data



"""
Funzione che, preso in input il dataset utilizzato per l'addestramento, elimina
le stopwords ed effettua la lemmatizzazione
"""
def stopWord_lemmatization(data):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')
    nlp = spacy.load("en_core_web_sm")

    # Tokenizzazione dei record
    for i,row in data.iterrows():
        tokens = tokenizer.tokenize(str(row['Text']))

        tokens_f = []
        for token in tokens:
            if(token.lower() not in stop_words):
                tokens_f.append(token)

        # Trasformazione lista token in stringhe
        doc = ' '.join(tokens_f)
        lem = nlp(doc)
        tokens_lem = []
        for token in lem:
            tokens_lem.append(token.lemma_)

        aus = (' ').join(tokens_lem)
        data.loc[i, 'Text'] = aus



"""
Funzione che effettua il training del classificatore
"""
def train_classifier(data):
    X = data['Text']
    Y = data['Emotion']

    cv = CountVectorizer()
    X = cv.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.3)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print('Accuracy: {0}'.format(acc))

    pickle.dump(model, open(model_path, 'wb'))
    vec_file = 'vectorizer.pickle'
    pickle.dump(cv, open(vec_path, 'wb'))



def create_model():
    print('Cleaning data training...')
    data = cleaning(data_path)
    print('Deleting stop words and applying lemmatization...')
    stopWord_lemmatization(data)
    print('Creating model...')
    train_classifier(data)

    cv = pickle.load(open(vec_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
