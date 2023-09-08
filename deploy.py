#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[7]:


from PIL import Image
import re
import string
import nltk


# In[9]:


get_ipython().system('pip install spacy')


# In[11]:


get_ipython().system('pip install pydantic')


# In[12]:


get_ipython().system('pip install typing-extensions')


# In[19]:


import spacy


# In[13]:





# In[16]:


with open("C:/Users/yuktha/svm_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("C:/Users/yuktha/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# In[18]:


nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = text.lower()
    return text.strip()

def remove_punctuation(text):
    punctuation_free = "".join([i for i in text if i not in string.punctuation])
    return punctuation_free

def tokenization(text):
    tokens = re.split(' ', text)
    return tokens

def remove_stopwords(text):
    output = " ".join(i for i in text if i not in stopwords)
    return output

def lemmatizer(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sent = [token.lemma_ for token in doc if not token.text in set(stopwords)]
    return ' '.join(sent)


# In[21]:


st.title("Sentiment Analysis App")
st.markdown("By yuktha")
image = Image.open("C:/Users/yuktha/Desktop/Threads/threads.jpeg")
st.image(image, use_column_width=True)

st.subheader("Enter your text here:")
user_input = st.text_area("")

if user_input:
    user_input = clean_text(user_input)
    user_input = remove_punctuation(user_input)
    user_input = user_input.lower()
    user_input = tokenization(user_input)
    user_input = remove_stopwords(user_input)
    user_input = lemmatizer(user_input)

if st.button("Predict"):
    if user_input:
        text_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(text_vectorized)[0]
        st.header("Prediction:")
        if prediction == -1:
            st.subheader("The sentiment of the given text is: Negative")
        elif prediction == 0:
            st.subheader("The sentiment of the given text is: Neutral")
        elif prediction == 1:
            st.subheader("The sentiment of the given text is: Positive")
    else:
        st.subheader("Please enter a text for prediction.")


# In[ ]:




