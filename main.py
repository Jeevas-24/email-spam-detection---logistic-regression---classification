from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tensorflow
import pandas as pd
import streamlit as st

df = pd.read_csv('spam.csv')
print(df.head(5)) # returns first 5 rows & we can leave head as empty it will give first and last 5 rows
# print(df.shape) # returns no of rows and columns in (row,col)
df.drop_duplicates(inplace=True) # remove duplicates in place(changes made to the df itself)
# print(df.shape)
print(df.isnull().sum()) # returns the total number of null values in dataset
df['Category'] = df['Category'].replace(['ham','spam'],['Not Spam','Spam']) # replace the ham and spam with Not Spam and Spam
print(df.head())

# Split data set for Training and testing
x = df['Message']
y = df['Category']

(x_train,x_test,y_train,y_test) = train_test_split(x,y,test_size=0.2) # splits the data and assigns it into appropriate variables with split percentage as 20 (test_size=0.2)

# Converts text into numbers
cv = CountVectorizer(stop_words='english')
x_features = cv.fit_transform(x_train) # converts message train into numeric value, extracts the features to make the model learn

# creating model
model = MultinomialNB() # MultinomialNB is used for classifying the data
model.fit(x_features,y_train) # learns from the feature along with label (category_train)

# Test our model
x_test = cv.transform(message_test)
print(model.score(x_test,y_test))

# Predict data
# x_predict = cv.transform(['Congrats you won a lottery']).toarray()
# print(type(x_predict))
# print(model.predict(x_predict)) # predicts the mail

def predict(m):
  message_predict = cv.transform([m]).toarray()
  res = model.predict(message_predict)
  return res

# Streamlit web app

st.header('Spam Detector')

input = st.text_input('Enter the message')
if st.button('Validate'):
  output = predict(input)
  st.markdown(output)








