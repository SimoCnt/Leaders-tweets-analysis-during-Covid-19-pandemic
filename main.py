"""
@author: Salvatore Calderaro
@author: Simone Contini
"""

from create_tweets_datasets import scraper
from preprocessing_data import *
from topic_modeling import *
from create_emotion_detection_model import *



def print_menu():
    os.system('cls||clear')
    print('MENU')
    print('================================')
    print('1 - Data scraping')
    print('2 - Analyze Italian Tweets')
    print('3 - Analyze England Tweets')
    print('4 - Analyze French Tweets')
    print('5 - Analyze Brasilian Tweets')
    print('6 - Analyze Indian Tweets')
    print('7 - Create emotion detection model')
    print('8 - Exit')
    print('================================')



"""
Menù principale
"""
def menu():
    print_menu()
    user_input = 0

    while user_input != 8:
        user_input = int(input())

        if user_input == 1:
            print('\nScraping data...')
            scraper()
            print('Operation complete!')

            while True:
                c = input("Press any key to continue")
                break

            print_menu()

        elif user_input == 2:
            for i in range(len(ita_tweets)):
                print('\nTweets period {0}'.format(i+1))
                print('Applying LDA...')
                vectorizer, lda_model, topic_indexes, n_topics = apply_LDA(ita_tweets[i])
                #print('Showing top 10 words per topic...')
                #show_top10words_topics(vectorizer, lda_model, 10)
                print('Generating Word Clouds...')
                create_wordCloud(topic_indexes, ita_tweets[i], n_topics, 'Italy', 'it', i+1)
                print('Generating emotions per topic...')
                detect_topic_emotion(topic_indexes, ita_tweets[i], n_topics, 'Italy', 'it', i+1)
                print('Showing results...')
                plot_analysis('Italy', i+1)
                detect_emotion_periods(ita_tweets[i], 'it', 'Italy', i+1)
                print('Operation complete!')

                while True:
                    c = input("Press any key to continue")
                    break

            print_menu()

        elif user_input == 3:
            for i in range(len(eng_tweets)):
                print('\nTweets period {0}'.format(i+1))
                print('Applying LDA...')
                vectorizer, lda_model, topic_indexes, n_topics = apply_LDA(eng_tweets[i])
                #print('Showing top 10 words per topic...')
                #show_top10words_topics(vectorizer, lda_model, 10)
                print('Generating Word Clouds...')
                create_wordCloud(topic_indexes, eng_tweets[i], n_topics, 'United-Kingdom', 'en', i+1)
                print('Generating emotions per topic...')
                detect_topic_emotion(topic_indexes, eng_tweets[i], n_topics, 'United-Kingdom', 'en', i+1)
                print('Showing results...')
                plot_analysis('United-Kingdom', i+1)
                detect_emotion_periods(eng_tweets[i], 'en', 'United-Kingdom', i+1)
                print('Operation complete!')

                while True:
                    c = input("Press any key to continue")
                    break

            print_menu()

        elif user_input == 4:
            for i in range(len(fra_tweets)):
                print('\nTweets period {0}'.format(i+1))
                print('Applying LDA...')
                vectorizer, lda_model, topic_indexes, n_topics = apply_LDA(fra_tweets[i])
                #print('Showing top 10 words per topic...')
                #show_top10words_topics(vectorizer, lda_model, 10)
                print('Generating Word Clouds...')
                create_wordCloud(topic_indexes, fra_tweets[i], n_topics, 'France', 'fr', i+1)
                print('Generating emotions per topic...')
                detect_topic_emotion(topic_indexes, fra_tweets[i], n_topics, 'France', 'fr', i+1)
                print('Showing results...')
                plot_analysis('France', i+1)
                detect_emotion_periods(fra_tweets[i], 'fr', 'France', i+1)
                print('Operation complete!')

                while True:
                    c = input("Press any key to continue")
                    break

            print_menu()

        elif user_input == 5:
            for i in range(len(bra_tweets)):
                print('\nTweets period {0}'.format(i+1))
                print('Applying LDA...')
                vectorizer, lda_model, topic_indexes, n_topics = apply_LDA(bra_tweets[i])
                #print('Showing top 10 words per topic...')
                #show_top10words_topics(vectorizer, lda_model, 10)
                print('Generating Word Clouds...')
                create_wordCloud(topic_indexes, bra_tweets[i], n_topics, 'Brasil', 'pt', i+1)
                print('Generating emotions per topic...')
                detect_topic_emotion(topic_indexes, bra_tweets[i], n_topics, 'Brasil', 'pt', i+1)
                print('Showing results...')
                plot_analysis('Brasil', i+1)
                detect_emotion_periods(bra_tweets[i], 'pt', 'Brasil', i+1)
                print('Operation complete!')

                while True:
                    c = input("Press any key to continue")
                    break

            print_menu()

        elif user_input == 6:
            for i in range(len(ind_tweets)):
                print('\nTweets period {0}'.format(i+1))
                print('Applying LDA...')
                vectorizer, lda_model, topic_indexes, n_topics = apply_LDA(ind_tweets[i])
                #print('Showing top 10 words per topic...')
                #show_top10words_topics(vectorizer, lda_model, 10)
                print('Generating Word Clouds...')
                create_wordCloud(topic_indexes, ind_tweets[i], n_topics, 'India', 'en', i+1)
                print('Generating emotions per topic...')
                detect_topic_emotion(topic_indexes, ind_tweets[i], n_topics, 'India', 'en', i+1)
                print('Showing results...')
                plot_analysis('India', i+1)
                detect_emotion_periods(ind_tweets[i], 'en', 'India', i+1)
                print('Operation complete!')

                while True:
                    c = input("Press any key to continue")
                    break

            print_menu()

        elif user_input == 7:
            print('\nCreating model...')
            create_model()
            print('Operation complete!')

            while True:
                c = input("Press any key to continue")
                break

            print_menu()

        elif user_input == 8:
            break



if __name__ == "__main__":
    menu()
