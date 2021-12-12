import tensorflow as tf
import numpy as np
import streamlit as st
import requests
import twitter
import tweepy
import pickle

#from PIL import Image
#from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer

#hide warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

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
model = tf.keras.models.load_model('../emotion-analysis/keras_metada.pb')


# load tokenizer for preprocessing
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


#function to get results for a particular text input     

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

    df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])
    t_h = twitter.Twitter(auth=twitter.OAuth2(bearer_token=BEARER_TOKEN))

    ## Function to extract tweets from twitter api
    def get_tweets(Topic,Count):
        i=0
        #my_bar = st.progress(100) # To track progress of Extracted tweets
        for tweet in tweepy.Cursor(t_h.search, q=Topic,count=100, lang="id",exclude='retweets').items():
            df.loc[i,"Date"] = tweet.created_at
            df.loc[i,"User"] = tweet.user.name
            df.loc[i,"IsVerified"] = tweet.user.verified
            df.loc[i,"Tweet"] = tweet.text
            df.loc[i,"Likes"] = tweet.favorite_count
            df.loc[i,"RT"] = tweet.retweet_count
            df.loc[i,"User_location"] = tweet.user.location
            i=i+1
            if i>Count:
                break
            else:
                pass
    
    ################################################################

    #Function to analyze sentiment
    def analyze_sentiment(tweet):
        analysis = model.predict([tweet])
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
        df['Sentiment'] = df['Tweet'].apply(lambda tweet: analyze_sentiment(tweet))

        # Write summary of the tweets
        st.write('Total length of the data is:         {}'.format(rnn_model.shape[0]))
        st.write('No. of sad tagged sentences is:  {}'.format(len(df[df['Sentiment']== 'sad' ])))
        st.write('No. of happy tagged sentences is: {}'.format(len(df[df['Sentiment'] == 'happy' ])))
        st.write('No. of angry tagged sentences is:  {}'.format(len(df[df['Sentiment']== 'angry' ])))
        st.write('No. of surprise tagged sentences is: {}'.format(len(df[df['Sentiment']== 'surprise' ])))
        st.write('No. of funny tagged sentences is:  {}'.format(len(df[df['Sentiment']== 'funny' ])))
        st.write('No. of hopeful tagged sentences is: {}'.format(len(df[df['Sentiment']== 'hopeful' ])))
        st.write('No. of disgust tagged sentences is:  {}'.format(len(df[df['Sentiment']== 'disgust'])))
        st.write('No. of neutral tagged sentences is: {}'.format(len(df[df['Sentiment']== 'neutral' ])))
        st.write('No. of fear tagged sentences is:  {}'.format(len(df[df['Sentiment']== 'fear' ])))

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
            a = len(df[df['Sentiment']== 'sad' ])
            b = len(df[df['Sentiment'] == 'happy' ])
            c = len(df[df['Sentiment']== 'angry' ])
            d = len(df[df['Sentiment']== 'surprise' ])
            e = len(df[df['Sentiment']== 'funny' ])
            f = len(df[df['Sentiment']== 'hopeful' ])
            g = len(df[df['Sentiment']== 'disgust' ])
            h = len(df[df['Sentiment']== 'neutral' ])
            i = len(df[df['Sentiment']== 'fear' ])

            
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
                            labels = ['sad', 'happy', 'angry', 'surprise', 'funny', 'hopeful', 'disgust', 'neutral', 'fear'], autopct='%1.2f%%'))
            st.pyplot()

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

