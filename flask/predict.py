import tensorflow as tf
import numpy as np
import streamlit as st
import requests
import twitter
import tweepy
import pickle
import pandas as pd
import re 

#from PIL import Image
#from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer


# import sys
# sys.path.append('..')
# from cust_class import EmotionDetector

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cust_class import EmotionDetector



#hide warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

# import warnings
#     warnings.filterwarnings('ignore')


#vizualization packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

# load the pipleline model
# model = tf.keras.models.load_model('model/emotion-analysis')


# load tokenizer for preprocessing
# with open('tokenizer.pickle', 'rb') as handle:
    # tokenizer = pickle.load(handle)


#function to get results for a particular text input     
df = pd.DataFrame()

def main():
    """ Common ML Dataset Explorer """

    html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment analysis</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Select a topic which you'd like to get the sentiment analysis on :")    

    ################# Twitter API Connection #######################
    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAI0YKQEAAAAAx7jZyYaOEmZo9rLnU%2BJ8NB%2FzM%2FA%3DSHlLkhfQLPc1ElDA60Uf2y2Ohl5rFNPYJJg3yXCj2UIX1LZ5SC'
    # consumer_key = lFbgvnDLVPb3bTg7eC3JiLsuK
    # consumer_secret = t6RuFpFDTaTdBG5PwP6q0QpgKOQsrZc792I54ydTqheBYXKBpB
    # access_token = 173410619-OmxC6NrKQOD9pztVtPGXeXdnKO9L6w4WK7Aoa9Uv
    # access_token_secret = tmGWwe0oCbcpFgggpJLaUIxfB9vVjufhzYiKwcpLAmOva

    
    t_h = twitter.Twitter(auth=twitter.OAuth2(bearer_token=BEARER_TOKEN))
    emotion_h = EmotionDetector.EmotionDetector()

    ## Function to extract tweets from twitter api
    def get_tweets(Topic,Count):
        global df
        is_full = False
        since_id_buff = ''
        i=0
        #my_bar = st.progress(100) # To track progress of Extracted tweets
        while len(df) < Count:
            print('.', end='')
            query = t_h.search.tweets(q=Topic, lang='id', count=100, tweet_mode='extended') if since_id_buff == '' else t_h.search.tweets(q=Topic, lang='id', count=100, tweet_mode='extended', max_id=since_id_buff)
            if "next_results" in query['search_metadata']:
                since_id_buff = re.search(r'max_id=(.+)&q', query['search_metadata']['next_results']).group(1)
            else:
                is_full = True
            query = pd.DataFrame(query['statuses'])

        # Get full text, put it in a new header
            query_rt = []
            if 'retweeted_status' in query.columns:
                for i in query['retweeted_status']:
                    try:
                        query_rt.append(i['full_text'])
                    except:
                        query_rt.append(np.nan)
            else:
                query_rt = [np.nan] * len(query)
            # print(query_rt)
            query_text = []
            for i in range(len(query_rt)):
            # print(i, end='\r')
                query_text.append(query_rt[i] if not pd.isnull(query_rt[i]) else query['full_text'].loc[i])

            query['inferred_text'] = query_text

        # Fixing
            df = df.append(query).reset_index().drop('index', axis=1) if df is not None else query

            if len(df) >= 300:
                df = df.drop_duplicates(subset=['inferred_text']).reset_index().drop('index', axis=1)

            #df.to_json('../dataset/trends/'+Topic+'.json', orient='index')
    print('Done!')
    
    ################################################################

    #Function to analyze sentiment
    def analyze_sentiment(tweet):
        analysis = emotion_h.predict_emotion(tweet)
        # analysis = model.predict([tweet])
        return analysis


    # Collect Input from user
    Topic = str()
    Topic = str(st.text_input("Enter Topic you are interested in (Press Enter Once done) :"))

    if len(Topic) > 0 :
        #call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner('Collecting Tweets...'):
            get_tweets(Topic, Count=200)
        st.success('Tweets have been collected !!!')

        # Load clean data
        # rnn_model = pd.read_json('../dataset/tweet-cleaned-5000.json', orient='index') 

        # Call Function to get the sentiment analysis
        df['Sentiment'] = df['inferred_text'].apply(lambda tweet: analyze_sentiment(tweet))

        # Write summary of the tweets
        #st.write('Total length of the data is:         {}'.format(rnn_model.shape[0]))
        st.write('No. of sad tagged sentences is:  {}'.format(len(df[df['Sentiment']== 'Sad' ])))
        st.write('No. of happy tagged sentences is: {}'.format(len(df[df['Sentiment'] == 'Happy' ])))
        st.write('No. of angry tagged sentences is:  {}'.format(len(df[df['Sentiment']== 'Angry' ])))
        st.write('No. of surprise tagged sentences is: {}'.format(len(df[df['Sentiment']== 'Surprise' ])))
        st.write('No. of funny tagged sentences is:  {}'.format(len(df[df['Sentiment']== 'Funny' ])))
        st.write('No. of hopeful tagged sentences is: {}'.format(len(df[df['Sentiment']== 'Hopeful' ])))
        st.write('No. of disgust tagged sentences is:  {}'.format(len(df[df['Sentiment']== 'Disgust'])))
        st.write('No. of neutral tagged sentences is: {}'.format(len(df[df['Sentiment']== 'Questioning' ])))
        st.write('No. of fear tagged sentences is:  {}'.format(len(df[df['Sentiment']== 'Fear' ])))

        # st.write('Total length of the data is:         {}'.format(rnn_model.shape[0]))
        # st.write('No. of sad tagged sentences is:  {}'.format(len(rnn_model['emotion'][rnn_model.emotion == 'sad' ])))
        # st.write('No. of happy tagged sentences is: {}'.format(len(rnn_model['emotion'][rnn_model.emotion == 'happy' ])))
        # st.write('No. of angry tagged sentences is:  {}'.format(len(rnn_model['emotion'][rnn_model.emotion == 'angry' ])))
        # st.write('No. of surprise tagged sentences is: {}'.format(len(rnn_model['emotion'][rnn_model.emotion == 'surprise' ])))
        # st.write('No. of funny tagged sentences is:  {}'.format(len(rnn_model['emotion'][rnn_model.emotion == 'funny' ])))
        # st.write('No. of hopeful tagged sentences is: {}'.format(len(rnn_model['emotion'][rnn_model.emotion == 'hopeful' ])))
        # st.write('No. of disgust tagged sentences is:  {}'.format(len(rnn_model['emotion'][rnn_model.emotion == 'disgust'])))
        # st.write('No. of neutral tagged sentences is: {}'.format(len(rnn_model['emotion'][rnn_model.emotion == 'neutral' ])))
        # st.write('No. of fear tagged sentences is:  {}'.format(len(rnn_model['emotion'][rnn_model.emotion == 'fear' ])))

        #piechart
        if st.button("Pie Chart for Different Sentiments"):
            st.success('Generating A Pie Chart...')
            a = len(df[df['Sentiment']== 'Sad' ])
            b = len(df[df['Sentiment'] == 'Happy' ])
            c = len(df[df['Sentiment']== 'Angry' ])
            d = len(df[df['Sentiment']== 'Surprise' ])
            e = len(df[df['Sentiment']== 'Funny' ])
            f = len(df[df['Sentiment']== 'Hopeful' ])
            g = len(df[df['Sentiment']== 'Disgust' ])
            h = len(df[df['Sentiment']== 'Questioning' ])
            i = len(df[df['Sentiment']== 'Fear' ])

            
            # a = len(rnn_model['emotion'][rnn_model.emotion == 'sad' ])
            # b = len(rnn_model['emotion'][rnn_model.emotion == 'happy' ])
            # c = len(rnn_model['emotion'][rnn_model.emotion == 'angry' ])
            # d = len(rnn_model['emotion'][rnn_model.emotion == 'surprise' ])
            # e = len(rnn_model['emotion'][rnn_model.emotion == 'funny' ])
            # f = len(rnn_model['emotion'][rnn_model.emotion == 'hopeful' ])
            # g = len(rnn_model['emotion'][rnn_model.emotion == 'disgust' ])
            # h = len(rnn_model['emotion'][rnn_model.emotion == 'neutral' ])
            # i = len(rnn_model['emotion'][rnn_model.emotion == 'fear' ])

            x = np.array([a,b,c,d,e,f,g,h,i])
            explode = (0, 0, 0, 0, 0, 0, 0, 0, 0)
            st.write(plt.pie(x,
                            explode=explode,
                            shadow = True,
                            labels =['Sad','Happy','Angry','Surprise','Funny','Hopeful','Disgust','Questioning','Fear'], autopct='%1.2f%%'))
            st.pyplot()

        # Predict Emotion


    st.sidebar.header("About App")
    st.sidebar.info("A Twitter Sentiment analysis Project which will scrap twitter for the topic selected by the user. The extracted tweets will then be used to determine the Sentiments of those tweets. \
                    The different Visualizations will help us get a feel of the overall mood of the people on Twitter regarding the topic we select.")
    st.sidebar.text("Built with Streamlit")
    
    st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
    st.sidebar.info("Alvin Januar & Aryo Anindyo")

    if st.button('Exit'):
        st.balloons()

if __name__ == '__main__':
    main()

