import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
from PIL import Image

#set title

st.title("Project_71:News Article Classification based on Headline and Body")

st.write("""#### Dataset """)
df=pd.read_json(r"C:\Users\Rathan\OneDrive\Documents\Docs\AppliedAI\Diploma\Project_71\Dataset\News_Category_Dataset_v3.json",lines=True)
st.dataframe(df)

st.write("""#### Categories of News Articles """)
category = pd.DataFrame(df['category'].value_counts()).reset_index()
st.bar_chart(category,x="index",y="category")

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
             "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", 
             "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", 
             "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", 
             "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
             "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", 
             "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours",
             "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
             "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
             "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
             "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where",
             "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you",
             "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stopwords:
            final_text.append(i.strip())
    return " ".join(final_text)
    
def alpha_num(text):
    return re.sub(r'[^A-Za-z0-9 ]', '', text)
    
    
def create_tf_matrix(category):
    return cvector.transform(df[df.category == category].headline)
    
    
def train_predict_model(classifier, 
                        train_features, train_labels, 
                        test_features, test_labels):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    return predictions   
    
def classify_utterance(utt):
    # load the vectorizer
    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('classification.model', 'rb'))

    # make a prediction
    print(loaded_model.predict(loaded_vectorizer.transform([utt])))

df['headline'] = df['headline'].apply(remove_stopwords)             
df['headline'] = df['headline'].str.lower()
df['headline'] = df['headline'].apply(alpha_num)

cvector = CountVectorizer(min_df = 0.0, max_df = 1.0, ngram_range=(1,2), stop_words = stopwords)
cvector.fit(df['headline'])

crime_matrix = create_tf_matrix('CRIME')
entertainment_matrix = create_tf_matrix('ENTERTAINMENT')
world_news_matrix = create_tf_matrix('WORLD NEWS')
impact_matrix = create_tf_matrix('IMPACT')
politics_matrix = create_tf_matrix('POLITICS')
weird_news_matrix = create_tf_matrix('WEIRD NEWS')
black_voices_matrix = create_tf_matrix('BLACK VOICES')
women_matrix = create_tf_matrix('WOMEN')
comedy_matrix = create_tf_matrix('COMEDY')
queer_voices_matrix = create_tf_matrix('QUEER VOICES')
sports_matrix = create_tf_matrix('SPORTS')
business_matrix = create_tf_matrix('BUSINESS')
travel_matrix = create_tf_matrix('TRAVEL')
media_matrix = create_tf_matrix('MEDIA')
tech_matrix = create_tf_matrix('TECH')
religion_matrix = create_tf_matrix('RELIGION')
science_matrix = create_tf_matrix('SCIENCE')
latino_voices_matrix = create_tf_matrix('LATINO VOICES')
education_matrix = create_tf_matrix('EDUCATION')
college_matrix = create_tf_matrix('COLLEGE')
parents_matrix = create_tf_matrix('PARENTS')
arts_and_culture_matrix = create_tf_matrix('ARTS & CULTURE')
style_matrix = create_tf_matrix('STYLE')
green_matrix = create_tf_matrix('GREEN')
taste_matrix = create_tf_matrix('TASTE')
healthy_living_matrix = create_tf_matrix('HEALTHY LIVING')
the_worldpost_matrix = create_tf_matrix('THE WORLDPOST')
good_news_matrix = create_tf_matrix('GOOD NEWS')
worldpost_matrix = create_tf_matrix('WORLDPOST')
fifty_matrix = create_tf_matrix('FIFTY')
arts_matrix = create_tf_matrix('ARTS')
wellness_matrix = create_tf_matrix('WELLNESS')
parenting_matrix = create_tf_matrix('PARENTING')
home_and_living_matrix = create_tf_matrix('HOME & LIVING')
style_and_beauty_matrix = create_tf_matrix('STYLE & BEAUTY')
divorce_matrix = create_tf_matrix('DIVORCE')
weddings_matrix = create_tf_matrix('WEDDINGS')
food_and_drink_matrix = create_tf_matrix('FOOD & DRINK')
money_matrix = create_tf_matrix('MONEY')
environment_matrix = create_tf_matrix('ENVIRONMENT')
culture_and_arts_matrix = create_tf_matrix('CULTURE & ARTS')

headline = np.array(df['headline'])
category = np.array(df['category'])

headline_train, headline_test, category_train, category_test = train_test_split(headline, category, test_size=0.2, random_state=42)

cv = CountVectorizer(stop_words='english',max_features=10000)
cv_train_features = cv.fit_transform(headline_train)

tv = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,2),
                     sublinear_tf=True)
tv_train_features = tv.fit_transform(headline_train)

cv_test_features = cv.transform(headline_test)
tv_test_features = tv.transform(headline_test)

lr = LogisticRegression(penalty='l2', max_iter=1000, C=1)
model=lr.fit(cv_train_features, category_train)

#lr_bow_predictions = train_predict_model(classifier=lr, 
#                                             train_features=cv_train_features, train_labels=category_train,
#                                             test_features=cv_test_features, test_labels=category_test)
                                             

vec_file = 'vectorizer.pickle'
pickle.dump(cv, open(vec_file, 'wb'))

# Save the model
mod_file = 'classification.model'
pickle.dump(model, open(mod_file, 'wb'))

st.subheader("Upload a query")
data_file=st.file_uploader("Upload CSV", type=["csv"])

if st.button("Process"):
    if data_file is not None:
        st.write(type(data_file))
        file_details = {"filename":data_file.name,"filetype":data_file.type,"filesize":data_file.size}
        st.write(file_details)
        query_df = pd.read_csv(data_file)
        st.write(query_df)
        st.write("Hello")
        st.write(classify_utterance(query_df['Query'][0].lower()))
        