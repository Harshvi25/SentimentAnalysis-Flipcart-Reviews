import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Load the data

df = pd.read_csv('D:\\Harshvi_MSCAIML\\AIML_Poject\\SentimentAnalysisFlipcartReviews\\archive\\Dataset-SA.csv').sample(frac = 0.1,random_state = 42)
print(df)

# Visualize the sentiment distribution

sentiment_count = df['Sentiment'].value_counts()
print(sentiment_count)
plt.bar(sentiment_count.index, sentiment_count.values)
plt.title("Sentiment Distribution")
plt.xlabel("Categories")
plt.ylabel("Frequency")
plt.show()

# Visualize customer reviews using WordCloud

''' Word cloud for summary column '''

txt = " ".join(df['Summary'].dropna())
wc = WordCloud(width = 800,height = 800,background_color = 'white').generate(txt)
plt.figure(figsize = (10,5))
plt.imshow(wc,interpolation = 'bilinear')
plt.title('Word Cloud of summary')
plt.show()

# Feature extraction

df1 = df.dropna()
x = df1[['Summary']]
y = df1[['Sentiment']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

# Vectorization of data

cv = CountVectorizer()
x_train_cnt = cv.fit_transform(x_train['Summary'])
x_test_cnt = cv.transform(x_test['Summary'])

# Train the model using MNB

mnb = MultinomialNB()
mnb.fit(x_train_cnt,y_train.values.ravel())

pred = mnb.predict(x_test_cnt)
acs = accuracy_score(y_test,pred)
print(acs)
cr = classification_report(y_test,pred)
print(cr)

# Train the model using RandomForest

rfc = RandomForestClassifier(n_estimators = 100,random_state = 42)
rfc.fit(x_train_cnt,y_train.values.ravel())

pred = rfc.predict(x_test_cnt)
acs = accuracy_score(y_test,pred)
print(acs)
cr = classification_report(y_test,pred)
print(cr)


# test the model using new data

def predict_sentiment_mt(txt):
    txt_counts = cv.transform([txt])
    prediction = mnb.predict(txt_counts)[0]
    return prediction

# case study

input_txt = "bad product i have erver seen"
predict_sentiment = predict_sentiment_mt(input_txt)
print("Predicted sentiment :",predict_sentiment)
