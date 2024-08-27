paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""

import re 
import nltk 
from nltk.corpus import stopwords 
from gensim.models import FastText 

def preprocess_text(func):
    def wrappers(paragraph, *args, **kwargs):
        txt = re.sub(r'\[[0-9]*\]', ' ', paragraph) 
        txt = re.sub(r'\s+', ' ', txt)
        txt = txt.lower() 
        txt = re.sub(r'\d', ' ', txt) 
        txt = re.sub(r'\s+', ' ', txt)
        return func(txt, *args, **kwargs)
    
    return wrappers 

@preprocess_text 
def prepare_text(text):
    sentences = nltk.sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    sents = [nltk.word_tokenize(sent) for sent in sentences]
    sents = [[word for word in sent if word not in stop_words] for sent in sents]
    return sents 


def train_model(sentences, sg, vector_size, window, epochs):
    model = FastText(sentences, sg = sg, vector_size = vector_size, window = window, min_count = 1, epochs = epochs)
    vocab = list(model.wv.index_to_key)
    return model, vocab 

def main(paragraph):
    text = prepare_text(paragraph) 

    model_cbow, vocab_cbow = train_model(text, sg = 0, vector_size = 50, window = 4, epochs =30)
    print("Vocab CBOW:", vocab_cbow) 

    vector_cbow = model_cbow.wv['world']
    print("Vector Cbow:", vector_cbow)

    model_skipgram, vocab_skipgram= train_model(text, sg = 1, vector_size= 50, window = 5, epochs = 30)
    print("Vocab Skipgram:", vocab_skipgram) 

    vector_skipgram = model_skipgram.wv['world']
    print("Vector Skipgram:", vector_skipgram) 

if __name__ == '__main__':
    main(paragraph)
