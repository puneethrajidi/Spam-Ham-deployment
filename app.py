import streamlit as st
st.title('Spam Ham Cassification')
import pandas as pd
df=pd.read_table('spam.tsv')
x=df['message']
y=df['label']
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
text_model=Pipeline([('tfid',TfidfVectorizer()),('model',SVC())])
text_model.fit(x,y)
select=st.text_input('Enter your msg')
op=text_model.predict([select])
st.title(op)
