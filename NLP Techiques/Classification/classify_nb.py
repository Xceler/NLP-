import re 
import nltk 
import pandas as pd 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

text_file = pd.read_csv("NLP Techiques/Classification/SMS.csv")

cor = [] 
wordnet = WordNetLemmatizer() 

for i in range(0, len(text_file)):
    review = re.sub('[^a-zA-Z]', ' ', text_file['Message'][i])
    review = review.lower() 
    txt = review.split() 
    txt = [wordnet.lemmatize(word) for word in txt if not word in set(stopwords.words('english'))]
    txt = ' '.join(txt)
    cor.append(txt)


cv = CountVectorizer() 
x = cv.fit_transform(cor).toarray()
y = pd.get_dummies(text_file['Labels'])
y = y.iloc[:, 1].values 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

multi_nb = MultinomialNB() 
model = multi_nb.fit(x_train, y_train)
y_pred = model.predict(x_test)
result = accuracy_score(y_pred, y_test)
result_1 = classification_report(y_pred, y_test)

print("Accuracy Score:", result)
print("Classification Report:", result_1) 