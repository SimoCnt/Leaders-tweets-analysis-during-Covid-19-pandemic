"""
@author: Salvatore Calderaro
@author: Simone Contini
"""

import twint
import pandas as pd
import os
import configobj

config = configobj.ConfigObj('env.b')

path = config['TWEETS_PATH']
ita_path = config['ITA_PATH']
eng_path = config['ENG_PATH']
fra_path = config['FRA_PATH']
bra_path = config['BRA_PATH']
ind_path = config['IND_PATH']



"""
Scraping dei tweet di Narendra Modi (India)
"""
def create_india_csv():
    c = twint.Config()
    username = "narendramodi"
    c.Username = username
    c.Since = '2020-01-01'
    c.Until = '2020-06-29'
    c.Limit = 3000
    c.Store_csv = True
    c.Output = path + "tweets_india_1.csv"
    twint.run.Search(c)

    c = twint.Config()
    username = "narendramodi"
    c.Username = username
    c.Since = '2020-06-30'
    c.Until = '2021-06-09'
    c.Limit = 3000
    c.Store_csv = True
    c.Output = path+"tweets_india_2.csv"
    twint.run.Search(c)

    data1 = pd.read_csv(path + "tweets_india_1.csv")
    data2 = pd.read_csv(path + "tweets_india_2.csv")
    all_data = [data1, data2]
    data = pd.concat(all_data)
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = data.sort_values(by='date')
    data = data[['username','date','tweet','language','hashtags']]
    data = data[data.language == 'en']
    os.remove(path + "tweets_india_1.csv")
    os.remove(path + "tweets_india_2.csv")
    data.to_csv(ind_path)



"""
Scraping dei tweet di Boris Johnson (Inghilterra)
"""
def create_england_csv():
    c = twint.Config()
    username = "BorisJohnson"
    c.Username = username
    c.Since = '2020-01-01'
    c.Until = '2021-06-09'
    c.Limit = 3000
    c.Store_csv = True
    c.Output = eng_path
    twint.run.Search(c)
    data = pd.read_csv(eng_path)
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = data[data.language == 'en']
    data = data.sort_values(by='date')
    data = data[['username','date','tweet','language','hashtags']]
    data.to_csv(eng_path)



"""
Scraping dei tweet di Jair Bolsonaro (Brasile)
"""
def create_brasil_csv():
    c = twint.Config()
    username = "jairbolsonaro"
    c.Username = username
    c.Since = '2020-01-01'
    c.Until = '2020-03-17'
    c.Limit = 3000
    c.Store_csv = True
    c.Output = path + "tweets_brasil_1.csv"
    twint.run.Search(c)

    c=twint.Config()
    username = "jairbolsonaro"
    c.Username = username
    c.Since = '2020-03-18'
    c.Until = '2021-06-09'
    c.Limit = 3000
    c.Store_csv = True
    c.Output = path + "tweets_brasil_2.csv"
    twint.run.Search(c)

    data1 = pd.read_csv(path + "tweets_brasil_1.csv")
    data2 = pd.read_csv(path + "tweets_brasil_2.csv")
    all_data = [data1,data2]
    data1.head()
    data2.head()
    data = pd.concat(all_data)
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = data.sort_values(by='date')
    data = data[['username', 'date', 'tweet', 'language', 'hashtags']]
    data = data[data.language == 'pt']
    os.remove(path + "tweets_brasil_1.csv")
    os.remove(path + "tweets_brasil_2.csv")
    data.to_csv(bra_path)



"""
Scraping dei tweet di Emmanuel Macron (Francia)
"""
def create_france_csv():
    c = twint.Config()
    username = "EmmanuelMacron"
    c.Username = username
    c.Since = '2020-01-01'
    c.Until = '2021-06-09'
    c.Limit = 3000
    c.Store_csv = True
    c.Output = fra_path
    twint.run.Search(c)
    data = pd.read_csv(fra_path)
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = data.sort_values(by='date')
    data = data[data.language == 'fr']
    data = data[['username','date','tweet','language','hashtags']]
    data.to_csv(fra_path)



"""
Scraping dei tweet di Giuseppe Conte & Palazzo Chigi (Italia)
"""
def create_italy_csv():
    c = twint.Config()
    username = "GiuseppeConteIT"
    c.Username = username
    c.Since = '2020-01-01'
    c.Until = '2021-02-13'
    c.Limit = 3000
    c.Store_csv = True
    c.Output = path + "tweets_italia_1.csv"
    twint.run.Search(c)

    c = twint.Config()
    username = "Palazzo_Chigi"
    c.Username = username
    c.Since = '2021-02-14'
    c.Until = '2021-06-09'
    c.Limit = 3000
    c.Store_csv = True
    c.Output = path + "tweets_italia_2.csv"
    twint.run.Search(c)
    data1 = pd.read_csv(path + "tweets_italia_1.csv")
    data2 = pd.read_csv(path + "tweets_italia_2.csv")
    all_data = [data1,data2]
    data = pd.concat(all_data)
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = data.sort_values(by='date')
    data = data[['username','date','tweet','language','hashtags']]
    data = data[data.language == 'it']
    os.remove(path + "tweets_italia_1.csv")
    os.remove(path + "tweets_italia_2.csv")
    data.to_csv(ita_path)



def scraper():
    create_india_csv()
    create_england_csv()
    create_brasil_csv()
    create_italy_csv()
    create_france_csv()
