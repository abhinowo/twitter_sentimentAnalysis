import pickle
import tensorflow as tf
import re
import emoji
import string
import numpy as np
import os 

class EmotionDetector:
    def __init__(self):
        print(os.getcwd()) 
        with open('wwt-emotion-master/model/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.model = tf.keras.models.load_model('wwt-emotion-master/model/emotion-analysis')
        self.model.summary()
        self.punctuation_gensim = re.sub('([.,?!])','', string.punctuation)
        self.max_length = 40
        self.trunc_type = 'post'
        self.padding = 'post'
        self.columns = ['Anger', 'Disgust', 'Fear', 'Happy', 'Hopeful', 'Surprise', 'Questioning', 'Sad', 'Laughter']

    def remove_twitter(self, text, for_gensim = False):
        twitter_stripped_text = re.sub('(@[A-Za-z0-9_]+)', '', text) if for_gensim else re.sub('(@)', '', text)   # Username removal
        twitter_stripped_text = re.sub(r'([http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*))', '', twitter_stripped_text)   # Link removal
        twitter_stripped_text = re.sub('(\#[A-Za-z0-9_]+)', '', twitter_stripped_text) if for_gensim else re.sub('(\#)', ' ', twitter_stripped_text)  # Hashtag removal, but preserving the word
        twitter_stripped_text = re.sub('([-&])', ' ', twitter_stripped_text)   # dash spacing, accommodating for better representation in spams
        return twitter_stripped_text

    def remove_punctuation(self, text, for_gensim=False):
        clean_text = "".join([i for i in text if i not in self.punctuation_gensim]) if for_gensim else "".join([i for i in text if i not in string.punctuation])
        clean_text = re.sub('([’])', "'", clean_text)
        return clean_text

    def remove_emoji(self, text):
        return re.sub(emoji.get_emoji_regexp(), r"", text)

    def preprocess(self, text):
        text = self.remove_twitter(text, for_gensim=True)
        text = self.remove_punctuation(text, for_gensim=True)
        text = self.remove_emoji(text).lower()

        # padded stuff
        seq = self.tokenizer.texts_to_sequences([text])
        pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=self.max_length,
            padding=self.padding, truncating=self.trunc_type)
        return pad

    def predict_emotion(self, text):
        pad_text = self.preprocess(text)
        return self.columns[np.argmax(self.model.predict([pad_text]))]