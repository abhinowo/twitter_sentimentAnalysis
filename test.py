from cust_class import EmotionDetector
import warnings

warnings.filterwarnings('ignore')

emotion_h = EmotionDetector.EmotionDetector()

sample_text = 'IHHHH GILA SENENG BANGET GUA ADA DIA DATENG HHHHH'

print('Sample text:', sample_text)

print('Detected emotion:', emotion_h.predict_emotion(sample_text))



