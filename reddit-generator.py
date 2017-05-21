# type chcp 65001 in windows terminal so that it doesnt give you unicode error
# coding: utf-8

# In[1]:

import numpy as np
import keras


# In[2]:

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import random
import sys
import re
import json


# In[3]:

def file_to_string(filename):
    data = ''
    
    with open(filename, 'r') as file:
        data = file.read()
        
    return data


# In[4]:

def split_into_sentences(data):
    sentences = re.split("(?<=}),", data)
    del sentences[-1] # remove empty string at the end
    return sentences


# In[5]:

def convert_to_json(data):
    jsonified = []
    
    for i in data:
        a = json.loads(i)
        jsonified.append(a)
        
    return jsonified


# In[36]:

def filter_titles(data,maxlen,score=1):
    high_score_titles = []
    
    for i in data:
        if i["score"] >= score and len(i["title"]) >= maxlen:
            high_score_titles.append(i["title"])
            
    return high_score_titles


# In[35]:

def find_longest_shortest_title(data):
    minlen = min(data, key=len)
    maxlen = max(data, key=len)
    
    return len(maxlen), len(minlen)


# In[41]:

def find_avg_title_length(corpus, array):
    
    return int(len(corpus)/len(array))


# In[8]:

def get_final_corpus(data):
    corpus = ''
    
    for i in data:
        corpus += i
    
    return corpus


# In[9]:

def get_unique_chars(data):
    return sorted(list(set(data)))


# In[10]:

def clean_corpus(exp, data):
    
    return re.sub(exp,' ',data)


# In[11]:

def create_char_indices(chars):
    
    char_indices = dict((c,i) for i,c in enumerate(chars))
    indices_char = dict((i,c) for i,c in enumerate(chars))
    
    return char_indices, indices_char


# In[12]:

def split_into_chunks(array, maxlen, step, exp):
    
    sentences = []
    next_chars = []
    
    for t in range(0, len(array)-maxlen, step):
        sentences.append(array[t:t+maxlen])
        next_chars.append(array[t+maxlen])
            
    return sentences, next_chars


# In[73]:

def vectorization(sentences, maxlen, chars, char_indices, next_chars):
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i,t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
        
    return X, y


# In[15]:

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[18]:

file_as_string = file_to_string('lifeprotips.txt')


# In[22]:

print("Num of chars:",len(file_as_string))


# In[24]:

raw_titles = split_into_sentences(file_as_string)


# In[25]:

print("Number of titles: ",len(raw_titles))


# In[26]:

json_sentences = convert_to_json(raw_titles)


# In[27]:

print("Num of json objects:", len(json_sentences))


# In[37]:
maxlen = 20
step = 3

score = 1
valid_titles = filter_titles(json_sentences, maxlen, score=score)


# In[38]:

print("Titles above score {}: {}".format(score, len(valid_titles)))


# In[39]:

longest_title, shortest_title = find_longest_shortest_title(valid_titles)


# In[40]:

print("Longest title:", longest_title)
print("Shortest title:", shortest_title)


# In[42]:

final_corpus = get_final_corpus(valid_titles)


# In[44]:

print("Length of final corpus:",len(final_corpus))


# In[45]:

avg_title_length = find_avg_title_length(final_corpus, valid_titles)


# In[46]:

print("Average Title Length:", avg_title_length)


# In[53]:

allchars = get_unique_chars(final_corpus)


# In[54]:

print("All chars", allchars)


# In[59]:

print("Length of all chars:", len(allchars))


# In[65]:

exp = '[\u200b|\u200e|\ufeff|\x7f]'
cleaned_corpus = clean_corpus(exp, final_corpus)


# In[66]:

print("Length of cleaned corpus:", len(cleaned_corpus))


# In[67]:

chars = get_unique_chars(cleaned_corpus)


# In[68]:

print("Length of Cleaned chars:", len(chars))


# In[69]:

char_indices, indices_char = create_char_indices(chars)


# In[70]:



sentences, next_chars = split_into_chunks(cleaned_corpus, maxlen, step, exp)


# In[71]:

print("Number of sentence chunks:", len(sentences))


# In[72]:

print("Number of next chars", len(next_chars))


# In[74]:

X, y = vectorization(sentences, maxlen, chars, char_indices, next_chars)


# In[75]:

print("Shape of training inputs:", X.shape)


# In[76]:

print("Shape of training labels", y.shape)


# In[80]:

def build_model(maxlen, chars):
    model = Sequential()

    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                 optimizer=optimizer)
    return model


# In[86]:

def train(model, X, y, seed, maxlen, length, ep):
    for i in range(1,ep):
        print()
        print('-'*50)
        print("Iteration",i)
        model.fit(X, y, batch_size=128, epochs=1)

        for temp in [0.2, 0.5, 1.0, 1.2]:
            print()
            print("----Temperature", temp)

            generated = ''
            sentence = seed
            generated+=sentence
            print("Using seed: ", seed)
            sys.stdout.write(generated)

            for i in range(length):
                x = np.zeros((1, maxlen, len(chars)))
                for t,char in enumerate(sentence):
                    x[0,t, char_indices[char]]=1

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, temp)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:]+next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
        filename = 'lifeprotips_model.h5'
        model.save(filename)
            


# In[81]:

model = build_model(maxlen, chars)


# In[85]:

seed = 'LPT: if you need to '
length = 180
ep = 25


# In[ ]:

train(model, X, y, seed, maxlen, length, ep)

