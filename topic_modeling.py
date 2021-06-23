"""
@author: Salvatore Calderaro
@author: Simone Contini
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation as LDA
from textblob import TextBlob
from wordcloud import WordCloud
from deep_translator import GoogleTranslator
import os
import configobj
import numpy as np
import pyLDAvis.sklearn
import pickle
import matplotlib.pyplot as plt
from PIL import Image

config = configobj.ConfigObj('env.b')

vec_path = config['MODEL_VEC_PATH']
model_path = config['MODEL_PATH']
wordCloud_path = config['WORDCLOUD_PATH']
topics_per_period_path = config['TOPICS_PER_PERIOD_PATH']
emotions_path = config['EMOTION_PATH']
emotions_per_topic_path = config['EMOTION_PER_TOPIC_PATH']
epidemic_path = config['EPIDEMIC_PATH']

cv = pickle.load(open(vec_path, 'rb')) # CountVectorizer
model = pickle.load(open(model_path, 'rb'))



"""
Funzione che applica l'algoritmo LDA (Latent Dirichlet Allocation) ai documenti
della lista in input.
"""
def apply_LDA(documents):
    count_vectorizer = CountVectorizer(min_df=10, max_df=0.95, ngram_range=(1,1))
    feature_matrix = count_vectorizer.fit_transform(documents)
    search_params = {'n_components': [5], 'learning_decay': [.5, .7, .9]}
    lda = LDA()
    model = GridSearchCV(lda, search_params)
    model.fit(feature_matrix)
    best_lda_model = model.best_estimator_
    lda_output = best_lda_model.transform(feature_matrix)
    prob_topic = np.round(lda_output, 2)
    topic_indexes = np.argmax(prob_topic, axis=1)
    n_topics = len(np.unique(topic_indexes))

    # Calcolo Perplexity (minore è, meglio è)
    #print('Perplexity: ', best_lda_model.perplexity(feature_matrix))

    # Calcolo Log likelihood (maggiore è, meglio è)
    #print('Log likelihood: ', best_lda_model.score(feature_matrix))

    return count_vectorizer, best_lda_model, topic_indexes, n_topics



"""
Funzione che stampa, per ogni topic, le top 10 word
"""
def show_top10words_topics(vectorizer, lda_model, n_words):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []

    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(list(keywords.take(top_keyword_locs)))

    print(topic_keywords)



"""
Funzione che crea una word cloud per i topic presi in input
"""
def create_wordCloud(topic_indexes, tweets, n_topics, country, lang, period):
    list_phrases_topics = [[] for _ in range(n_topics)]
    i = 0
    for topic in topic_indexes:
        list_phrases_topics[topic].append(tweets[i])
        i += 1

    for j in range(len(list_phrases_topics)):
        text_cloud = ''

        for feedback in list_phrases_topics[j]:
            text_cloud += ' ' + feedback

        cloud = WordCloud(max_font_size = 40).generate(text_cloud)

        path = wordCloud_path + country + '/' + country + '_' + str(period) + '_' + 'topic' + str(j) + '.png'
        cloud.to_file(path)



"""
Funzione che, per il periodo preso in input, effettua l'emotion analysis di tutti
i topic.
"""
def detect_topic_emotion(topic_indexes, tweets, n_topics, country, lang, period):
    list_phrases_topics = [[] for _ in range(n_topics)]
    i = 0
    for topic in topic_indexes:
        list_phrases_topics[topic].append(tweets[i])
        i += 1

    for j in range(len(list_phrases_topics)):
        freq = {"joy": 0, "sadness": 0, "fear": 0, "anger": 0, "surprise": 0, "neutral": 0, "disgust": 0, "shame": 0}
        for topic_tweets in list_phrases_topics[j]:
            if lang != 'en':
                topic_tweets = GoogleTranslator(source='auto', target='en').translate(topic_tweets)

            aus = []
            aus.append(topic_tweets)
            tweet_vec = cv.transform(aus).toarray()
            res = model.predict(tweet_vec)
            freq[res[0]] += 1

        labels = freq.keys()
        x_pos = np.arange(len(labels))
        val = freq.values()

        fig_title = country + ' ' + str(period) + ' ' + 'Topic' + ' ' + str(j+1) + ' ' + 'Emotion Analysis'
        fig_emo_topic = plt.figure()
        plt.title(fig_title, fontsize=12)
        plt.pie(val)
        plt.legend(labels, loc="best")
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(emotions_per_topic_path + country + '/' + str(period) + '/' + country + '_' + str(j+1) + '.png', format='png', dpi=1200)
        plt.close(fig_emo_topic)



"""
Funzione che, per il periodo preso in input, effettua l'emotion analysis (attraverso
l'uso di algoritmi di machine learning) dei tweet
"""
def detect_emotion_periods(tweets, lang, country, period):
    freq = {"joy": 0, "sadness": 0, "fear": 0, "anger": 0, "surprise": 0, "neutral": 0, "disgust": 0, "shame": 0}

    for tweet in tweets:
        if lang != 'en':
            text = GoogleTranslator(source='auto', target='en').translate(tweet)
        else:
            text = tweet

        aus = []
        aus.append(text)
        text_vec = cv.transform(aus).toarray()

        res = model.predict(text_vec)
        freq[res[0]] += 1

    if period == 1:
        per = '01/01/2020 - 31/05/2020'
    elif period == 2:
        per = '01/06/2020 - 30/09/2020'
    elif period == 3:
        per = '01/10/2020 - 31/12/2020'
    else:
        per = '01/01/2021 - 09/06/2021'

    labels = freq.keys()
    x_pos = np.arange(len(labels))
    val = freq.values()

    fig_title = country + ' ' + per + ' ' + 'Emotion Analysis'
    fig_emo_topic2 = plt.figure()
    plt.title(fig_title, fontsize=12)
    plt.pie(val)
    plt.legend(labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(emotions_path + country + '/' + country + '_' + str(period) + '.png', format='png', dpi=1200)
    plt.show()



"""
Plot grafici
"""
def plot_analysis(country, period):
    freq = {"joy": 0, "sadness": 0, "fear": 0, "anger": 0, "surprise": 0, "neutral": 0, "disgust": 0, "shame": 0}
    labels = freq.keys()

    wc_image_path = wordCloud_path + country + '/'
    emo_per_topic_path = emotions_per_topic_path + country + '/' + str(period) + '/'
    emo_image_path = emotions_path + country + '/' + country + '_' + str(period) + '.png'
    epidemic_image_path = epidemic_path + country + '.png'

    if period == 1:
        per = '01/01/2020 - 31/05/2020'
    elif period == 2:
        per = '01/06/2020 - 30/09/2020'
    elif period == 3:
        per = '01/10/2020 - 31/12/2020'
    else:
        per = '01/01/2021 - 09/06/2021'

    all_files_wc = os.listdir(wc_image_path)
    all_files_pie = os.listdir(emo_per_topic_path)
    cur_wc_files = []
    cur_pie_files = []

    for file1 in (all_files_wc):
        if('_' + str(period) in file1):
            cur_wc_files.append(file1)

    cur_wc_files.sort(key=lambda x: x.split('_topic')[1])
    all_files_pie.sort(key=lambda x: x.split('_')[1])

    list_wordClouds_images = []
    for f in cur_wc_files:
        f = wc_image_path + f
        list_wordClouds_images.append(Image.open(f,'r'))

    list_piePerTopic_images = []
    for f in all_files_pie:
        f = emo_per_topic_path + f
        list_piePerTopic_images.append(Image.open(f,'r'))

    fig = plt.figure()
    gs1 = fig.add_gridspec(nrows=2,ncols=3)
    ax1 = fig.add_subplot(gs1[0,0])
    ax2 = fig.add_subplot(gs1[0,1])
    ax3 = fig.add_subplot(gs1[0,2])
    ax4 = fig.add_subplot(gs1[1,0])
    ax5 = fig.add_subplot(gs1[1,1])
    ax6 = fig.add_subplot(gs1[1,2])

    ax1.imshow(list_wordClouds_images[0])
    ax1.set_title("Topic 1")
    ax1.axis('off')

    ax2.imshow(list_wordClouds_images[1])
    ax2.set_title("Topic 2")
    ax2.axis('off')

    ax3.imshow(list_wordClouds_images[2])
    ax3.set_title("Topic 3")
    ax3.axis('off')

    ax4.imshow(list_piePerTopic_images[0])
    ax4.set_title("Emotion Pie Topic 1")
    ax4.axis('off')

    ax5.imshow(list_piePerTopic_images[1])
    ax5.set_title("Emotion Pie Topic 2")
    ax5.axis('off')

    ax6.imshow(list_piePerTopic_images[2])
    ax6.set_title("Emotion Pie Topic 3")
    ax6.axis('off')

    sub_title = 'List of topics ' + per + ' - ' + country
    fig.suptitle(sub_title)
    plt.savefig(topics_per_period_path + country + '/' + country + '_' + str(period) + '1.png', format='png', dpi=1200)
    plt.show()

    fig2 = plt.figure()
    gs2 = fig2.add_gridspec(nrows=2,ncols=2)
    ax1 = fig2.add_subplot(gs2[0,0])
    ax2 = fig2.add_subplot(gs2[0,1])
    ax3 = fig2.add_subplot(gs2[1,0])
    ax4 = fig2.add_subplot(gs2[1,1])

    ax1.imshow(list_wordClouds_images[3])
    ax1.set_title("Topic 4")
    ax1.axis('off')

    ax2.imshow(list_wordClouds_images[4])
    ax2.set_title("Topic 5")
    ax2.axis('off')

    ax3.imshow(list_piePerTopic_images[3])
    ax3.set_title("Emotion Pie Topic 4")
    ax3.axis('off')

    ax4.imshow(list_piePerTopic_images[4])
    ax4.set_title("Emotion Pie Topic 5")
    ax4.axis('off')

    sub_title = 'List of topics ' + per + ' - ' + country
    fig2.suptitle(sub_title)
    plt.savefig(topics_per_period_path + country + '/' + country + '_' + str(period) + '2.png', format='png', dpi=1200)
    plt.show()
