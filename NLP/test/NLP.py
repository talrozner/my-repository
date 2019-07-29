#%%import
import numpy as np
import pandas as pd


#%% load data
#file_path = r"D:\DS\NLP\toxic-comments-classification\data\train.csv"
#data = pd.read_csv(file_path)
#data.set_index('id',drop = True,inplace = True)

#Text Preprocessing
#%%Text Preprocessing - Noise Removal

#from googletrans import Translator
#translator = Translator()
#translator.translate('안녕하세요.')
#
#translator.translate('text')
#
#from translate import Translator
#translator= Translator(to_lang="zh")
#translation = translator.translate("This is a pen.")
#
#from google.cloud import translate
#client = translate.Client()
#client.translate('koszula')

#%%
import spacy
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
#from contractions import CONTRACTION_MAP
import unicodedata

nlp = spacy.load('en', parse=True, tag=True, entity=True)
nltk.download('stopwords')
#nlp_vec = spacy.load('en_vecs', parse = True, tag=True, #entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('arabic')#'arabic'#'english'
#stopword_list.remove('no')
#stopword_list.remove('not')

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

text ="اتفق المجلس العسكري في السودان وتحالف قوى الحرية والتغيير المعارض، على ترتيبات هيكل السلطة الانتقالية في البلاد، وتحدد المرحلة الانتقالية بثلاث سنوات، وسيتم التوصل لاتفاق نهائي حول تفاصيل المرحلة خلال 24 ساعة."
#text = "The, and, if are stopwords, computer is not"
rsw = remove_stopwords(text)

from nltk.stem.wordnet import WordNetLemmatizer 
nltk.download('wordnet')
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word_list = text.split(" ")
for word in word_list:
    stem_results = stem.stem(word)
    print("word = " + word + "\n" + "stem = " + stem_results + "\n")
    
for word in word_list:
    lem_results = lem.lemmatize(word, "v")#not suppose to work
    print("word = " + word + "\n" + "lem = " + lem_results + "\n")



























