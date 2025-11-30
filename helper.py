import numpy as np
import pandas as pd
import re
import string
import nltk
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer() # initialize the PorterStemmer

#load model
with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

#load stopwords
with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()  # read stopwords from the downloaded file

#load vocabulary
vocab = pd.read_csv('static/model/vocabulary.txt',header=None)
token = vocab[0].tolist()  # convert vocabulary dataframe to list

def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
 #check punctuation in the each row and remove them and place with empty string ' '

def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split())) # convert to lowercase
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x,flags=re.MULTILINE) for x in x.split())) # remove URLs
    data["tweet"] = data["tweet"].apply(remove_punctuation) # remove punctuation from the tweets
    data["tweet"] = data['tweet'].str.replace(r'\d+', '', regex=True) # remove numbers from the tweets
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sw)) # remove stopwords from the tweets
    data['tweet'] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split())) # apply stemming to the tweets

    return data['tweet']


def vectorizer(ds):
    vectorized_1st = []  
    for sentence in ds:   
        sentence_1st = np.zeros(len(token)) 
        for i in range(len(token)):
            if token[i] in sentence.split():
                sentence_1st[i] = 1
        vectorized_1st.append(sentence_1st)  
    vectorized_1st_new = np.asarray(vectorized_1st, dtype=np.float32)
    return vectorized_1st_new  


def get_prediction(vectorized_txt):
    prediction = model.predict(vectorized_txt)
    if prediction ==1:
        return "negative"
    else:
        return "positive"
    