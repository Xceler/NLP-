import csv 

input_fil = "NLP Techiques/Classification/SMSSpamCollection.txt"
output_fil = "NLP Techiques/Classification/SMS.csv"

delimiter = '\t'

with open(input_fil, 'r') as txt_file, open(output_fil, 'w', newline = '') as csv_file:
    reader = csv.reader(txt_file, delimiter = delimiter)
    writer = csv.writer(csv_file)

    writer.writerow(["Message", "Labels"])

    for row in reader:
        if len(row) == 2:
            writer.writerow(row)


print(f"Converted From {input_fil} To {output_fil}")

import pandas as pd 

txt_fil = pd.read_csv("NLP Techiques/Classification/SMS.csv")


import re 
import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 


wordnet = WordNetLemmatizer() 
corpus = [] 

for i in range(0, len(txt_fil)):
    txt = re.sub('[^a-zA-Z]', ' ', txt_fil['Message'][i])
    txt  = txt.lower() 
    txt = txt.split() 
    txt = [wordnet.lemmatize(word) for word in txt if not word in set(stopwords.words('english'))]
    txt = ' '.join(txt)
    corpus.append(txt)


tfid = TfidfVectorizer() 
x = tfid.fit_transform(corpus).toarray() 
y = pd.get_dummies(txt_fil['Labels'])
y = y.iloc[:, 1].values 

x_train,x_test, y_train , y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

svm_model = SVC(kernel = 'linear')
model = svm_model.fit(x_train,y_train)

y_pred = model.predict(x_test)

result = accuracy_score(y_test, y_pred)
confusion_res = confusion_matrix(y_test, y_pred)

print("Accuracy Score:", result)
print("Confusion Matrix:", confusion_res)
