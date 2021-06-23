"""
@author: Salvatore Calderaro
@author: Simone Contini
"""

import pandas as pd
import numpy as np
import re
import os
import configobj
import spacy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

#pd.set_option('display.max_colwidth', -1)
#pd.set_option("max_rows", None)

config = configobj.ConfigObj('env.b')

ita_path = config['ITA_PATH']
eng_path = config['ENG_PATH']
fra_path = config['FRA_PATH']
bra_path = config['BRA_PATH']
ind_path = config['IND_PATH']

# Spacy configuration
nlp_ita = spacy.load("it_core_news_sm")
nlp_eng = spacy.load("en_core_web_sm")
nlp_fra = spacy.load("fr_core_news_sm")
nlp_bra = spacy.load("pt_core_news_sm")

# Stop words configuration
stop_words_ita = stopwords.words('italian')
stop_words_eng = stopwords.words('english')
stop_words_fra = stopwords.words('french')
stop_words_bra = stopwords.words('portuguese')



"""
Funzione che prende in input un dataset di tweets ed effettua il cleaning dei
dati
"""
def cleaning(path):
    data = pd.read_csv(path)

    for i,row in data.iterrows():
        row['tweet'] = re.sub('\S*@\S*\s?', '', str(row['tweet']))  # Rimozione Emails
        row['tweet'] = re.sub('\s+', ' ', str(row['tweet']))  # Rimozione new line
        row['tweet'] = re.sub("\'", "", str(row['tweet']))  # Rimozione punteggiatura
        row['tweet'] = re.sub(r'http\S+', ' ', str(row['tweet']))  # Rimozione links
        row['tweet'] = row['tweet'].lower()    # Conversione caratteri in minuscolo

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
        row['tweet'] = emoji_pattern.sub(r'', row['tweet'])  # Rimozione emoticons

        data.loc[i, 'tweet'] = row['tweet']

    return data



def clean_all_data():
    italia_df = cleaning(ita_path)
    england_df = cleaning(eng_path)
    france_df = cleaning(fra_path)
    brasil_df = cleaning(bra_path)
    india_df = cleaning(ind_path)

    return italia_df, england_df, france_df, brasil_df, india_df



"""
Restitusce una lista contenente 4 dataframe contenenti i tweets di un esponente
politico per ciascuno dei 4 periodi considerati:

    Primo periodo: 1 Gennaio 2020 - 31 Maggio 2020
    Secondo periodo: 1 Giugno 2020 - 30 Settembre 2020
    Terzo periodo: 1 Ottobre 2020 - 31 Gennaio 2021
    Quarto periodo: 1 Febbraio 2021 - Oggi
"""
def split_periods(data):
    per1_start_date = '2020-01-01'
    per1_end_date = '2020-05-31'
    per2_start_date = '2020-06-01'
    per2_end_date = '2020-09-30'
    per3_start_date = '2020-10-01'
    per3_end_date = '2021-01-31'
    per4_start_date = '2021-02-01'
    per4_end_date = '2021-06-09'

    after_per1_start_date = data['date'] >= per1_start_date
    after_per2_start_date = data['date'] >= per2_start_date
    after_per3_start_date = data['date'] >= per3_start_date
    after_per4_start_date = data['date'] >= per4_start_date
    before_per1_end_date = data['date'] <= per1_end_date
    before_per2_end_date = data['date'] <= per2_end_date
    before_per3_end_date = data['date'] <= per3_end_date
    before_per4_end_date = data['date'] <= per4_end_date

    between_per1_two_dates = after_per1_start_date & before_per1_end_date
    between_per2_two_dates = after_per2_start_date & before_per2_end_date
    between_per3_two_dates = after_per3_start_date & before_per3_end_date
    between_per4_two_dates = after_per4_start_date & before_per4_end_date

    per1_data = data.loc[between_per1_two_dates]
    per2_data = data.loc[between_per2_two_dates]
    per3_data = data.loc[between_per3_two_dates]
    per4_data = data.loc[between_per4_two_dates]
    periods=[per1_data, per2_data, per3_data, per4_data]

    return periods



"""
Funzione che, preso in input un dataset e relativa lingua dei tweet, elimina
le stopwords del linguaggio considerato ed effettua la lemmatizzazione
"""
def stopWord_lemmatization(data, lang):
    tokenizer = RegexpTokenizer(r'\w+')

    if lang == 'it':
        nlp = nlp_ita
        stop_words = stop_words_ita
    elif lang == 'en':
        nlp = nlp_eng
        stop_words = stop_words_eng
    elif lang == 'fr':
        nlp = nlp_fra
        stop_words = stop_words_fra
    else:
        nlp = nlp_bra
        stop_words = stop_words_bra

    # Tokenizzazione dei record
    documents = []
    for i,row in data.iterrows():
        tokens = tokenizer.tokenize(str(row['tweet']))

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

        documents.append(' '.join(tokens_lem))

    return documents



"""
Funzione che applica rimozione delle stopword e lemmatizzazione di tutti i
dataset. I dataset finali sono restituiti in una lista, contentente 4 dataframe,
uno per ogni periodo considerato
"""
def prepare_documents(dataframes,lang):
    documents=[]
    for dataframe in dataframes:
        documents.append(stopWord_lemmatization(dataframe,lang))

    return documents



print('\n\nLoading and cleaning data...')
italia_df, england_df, france_df, brasil_df, india_df = clean_all_data()

print('Splitting data per period...')
periods_ita = split_periods(italia_df)
periods_eng = split_periods(england_df)
periods_fra = split_periods(france_df)
periods_bra = split_periods(brasil_df)
periods_ind = split_periods(india_df)

print('Stop words and lemmatization...')
ita_tweets = prepare_documents(periods_ita, 'it')
eng_tweets = prepare_documents(periods_eng, 'en')
fra_tweets = prepare_documents(periods_fra, 'fr')
bra_tweets = prepare_documents(periods_bra, 'pt')
ind_tweets = prepare_documents(periods_ind, 'en')
