import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import  pad_sequences
import wget as wget
import csv
import pdb

#wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \ -O /Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/Data/

base_dir='/Users/ashishbansal/PycharmProjects/TensorflowProject/Coursera/'
data_dir=base_dir+'Data/'
url='https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'
def download_data(url,data_dir):

    wget.download(url=url,out=data_dir)
#download_data(url,data_dir)

with open(data_dir+'sarcasm.json') as f:
    data=json.load(f)
url=[]
sentences=[]
labels=[]
for item in data:
     url.append(item['article_link'])
     sentences.append(item['headline'])
     labels.append(item['is_sarcastic'])

#print(type(data))

#print(sentences[0])
token=Tokenizer()
texts=token.fit_on_texts(sentences)
sequences =token.texts_to_sequences(sentences)
padded =pad_sequences(sequences=sequences,padding='post')
print(sequences[0:10])
print(padded[0:10])
#print(padded.shape)
###################################################################
#wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv  -O /Data/bbc-text.csv
stopwords=[ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


labels=[]
sentences=[]
with open(data_dir+'bbc-text.csv','r') as f:
    data=csv.reader(f,delimiter=',')
    next(data)
    for row in data :
        labels.append(row[0])
        sentence=row[1]
        for word in stopwords:
            token=" "+word+" "
            sentence=sentence.replace(token," ")
            sentence =sentence.replace("  "," ")
            # print(sentence)
        sentences.append(sentence)

print(len(sentences))
#print(sentences[0])

token=Tokenizer(oov_token="<OOV>")
token.fit_on_texts(sentences)
word_index=token.word_index
print(word_index)
print(len(word_index))
sequence =token.texts_to_sequences(sentences)
padded=pad_sequences(sequences=sequence,padding='post')
print(padded.shape)
print(padded[0])




