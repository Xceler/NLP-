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
from gensim.models import Word2Vec 

def preprocess_text(func):
    def wrappers(paragraph, *args , ** kwargs):
        txt = re.sub(r'\[[0-9]*\]',  ' ', paragraph)
        txt = re.sub(r'\s+', ' ', txt)
        txt = txt.lower() 
        txt = re.sub(r'\d', ' ', txt) 
        txt = re.sub(r'\s+', ' ', txt)
        return func(txt, *args, **kwargs)
    
    return wrappers 


def prepare_text(text):
    sentences = nltk.sent_tokenize(text) 
    stop_words = set(stopwords.words('english'))
    sents = [nltk.word_tokenize(sent) for sent in sentences] 
    sents = [[word for word in sent if word not in stop_words] for sent in sents] 
    return sents 

""" 
      Model is Word2Vec (Word Embedding Model) 
      Technique is CBOW 
                                              
                                              """
                    

def train_model_cbow(sentences, min_count = 1, sg = 0):
    model = Word2Vec(sentences, min_count = min_count, sg = sg) 
    vocab = model.wv.index_to_key
    return model, vocab 


def main_cbow(paragraph):
    text = prepare_text(paragraph) 
    model, vocab = train_model_cbow(text) 

    vector = model.wv['world'] 
    print("Vector From CBOW:", vector)


"""    
        Model is Word2Vec (Word Embedding Model)
        Technique is Skipgram                    
                                                  """ 


def train_model_skipgram(sentences, min_count = 1, sg = 1):
    model = Word2Vec(sentences, min_count = min_count ,sg = sg) 
    vocab = model.wv.index_to_key 
    return model, vocab 

def main_skipgram(paragarph):
    text = prepare_text(paragraph) 
    model, vocab = train_model_skipgram(text) 

    vector = model.wv['world']

    print("Vector From SkipGram:", vector) 


if __name__ == '__main__':
    main_cbow(paragraph) 
    main_skipgram(paragraph) 