import pandas as pd
file_path = r"D:\DS\NLP\NLP_PIPLINE\Tweets.csv\Tweets.csv"
data = pd.read_csv(file_path)
#--Text Preprocessing
#%%--Text Preprocessing -- Noise Removal

import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 

def clean_text(text):
    text = re.sub(r"[.]",'',text)
    text = re.sub(r"[@][a-zA-Z0-9]+",'',text)
    text = text.lower()
    text = [word for word in text.split(' ') if not word in stop_words]
    text = " ".join(text)
    return text

text = r"@VirginAmerica plus you've added commercials to the experience... tacky."

new_text = clean_text(text)

print("clean text: " +"\n" + str(new_text))

#--Text Preprocessing ----Lexicon Normalization
#%%--Text Preprocessing ----Lexicon Normalization -- Stemming
from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word = "multiplying" 

print('\n'+"original word: "+'\n'+word+'\n'+'\n'+"stem word: "+"\n"+  stem.stem(word))

#%%--Text Preprocessing ----Lexicon Normalization -- Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

print('\n'+'lemma word with pos v:' +"\n"+  lem.lemmatize(word, "v"))

#%%--Text Preprocessing ----Lexicon Normalization -- Lemmatization -- Wordnet Lemmatizer with appropriate POS tag
import nltk
print('\n'+'word lemma pos :')
print(nltk.pos_tag([word]))

print('\n'+'text lemma pos :')
print(nltk.pos_tag(nltk.word_tokenize(new_text)))

#%%--Text Preprocessing ----Lexicon Normalization --Lemmatization -- Lemmatize with POS Tag
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


#%%--Text Preprocessing ----Lexicon Normalization --Lemmatization -- Lemmatize with POS Tag -- 1. Init Lemmatizer
lemmatizer = WordNetLemmatizer()

#%%--Text Preprocessing ----Lexicon Normalization --Lemmatization -- Lemmatize with POS Tag -- 2. Lemmatize Single Word with the appropriate POS tag
print('\n'+'Lemmatize Single Word with the appropriate POS tag:')
print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))

#%%--Text Preprocessing --Lexicon Normalization --Lemmatization -- Lemmatize with POS Tag -- 3. Lemmatize a Sentence with the appropriate POS tag
print('\n'+"original text: "+'\n' + new_text)
print('\n'+'Lemmatize a Sentence with the appropriate POS tag:')
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(new_text)])


#%%--Text Preprocessing -- Object Standardization
lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love"}

def _lookup_words(input_text):
    words = input_text.split() 
    new_words = [] 
    for word in words:
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
        new_words.append(word)
        new_text = " ".join(new_words) 
    return new_text

_lookup_words("RT this is a retweeted tweet by Shivam Bansal")


#-- Text to Features (Feature Engineering on text data)

#-- Text to Features -- Syntactic Parsing
#%%-- Text to Features -- Syntactic Parsing -- Dependency Trees
#Generating Dependency Trees using Stanford Core NLP
import spacy

nlp = spacy.load("en")

doc = nlp(u'The team is not performing well in the match')

for token in doc:
    print (str(token.text),  str(token.lemma_),  str(token.pos_),  str(token.dep_))
    
#from spacy import displacy
#displacy.serve(doc, style='dep',page=False)

#%%-- Text to Features -- Syntactic Parsing -- Part of speech tagging

from nltk import word_tokenize, pos_tag
text = "I am learning Natural Language Processing on Analytics Vidhya"
tokens = word_tokenize(text)
print(pos_tag(tokens))

#%%-- Text to Features -- Syntactic Parsing -- Topic Modeling

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father." 
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc_complete = [doc1, doc2, doc3]
doc_clean = [doc.split() for doc in doc_complete]

import gensim 
from gensim import corpora

# Creating the term dictionary of our corpus, where every unique term is assigned an index.  
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above. 
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

# Results 
print(ldamodel.print_topics())

#%%-- Text to Features -- Syntactic Parsing -- N-Grams as Features
def generate_ngrams(text, n):
    words = text.split()
    output = []  
    for i in range(len(words)-n+1):
        output.append(tuple(words[i:i+n]))
    return output

text_ngrams = r'Feynman was born on May 11, 1918, in Queens, New York City,[2] to Lucille née Phillips, a homemaker, and Melville Arthur Feynman, a sales manager[3] originally from Minsk in Belarus [4] (then part of the Russian Empire). Both were Lithuanian Jews.[5] Feynman was a late talker, and did not speak until after his third birthday. As an adult he spoke with a New York accent[6][7] strong enough to be perceived as an affectation or exaggeration[8][9]—so much so that his friends Wolfgang Pauli and Hans Bethe once commented that Feynman spoke like a "bum".[8] The young Feynman was heavily influenced by his father, who encouraged him to ask questions to challenge orthodox thinking, and who was always ready to teach Feynman something new. From his mother, he gained the sense of humor that he had throughout his life. As a child, he had a talent for engineering, maintained an experimental laboratory in his home, and delighted in repairing radios. When he was in grade school, he created a home burglar alarm system while his parents were out for the day running errands.[10] When Richard was five his mother gave birth to a younger brother, Henry Phillips, who died at age four weeks.[11] Four years later, Richards sister Joan was born and the family moved to Far Rockaway, Queens.[3] Though separated by nine years, Joan and Richard were close, and they both shared a curiosity about the world. Though their mother thought women lacked the capacity to understand such things, Richard encouraged Joans interest in astronomy, and Joan eventually became an astrophysicist.[12]'

bigrams = generate_ngrams(text_ngrams, 2)

from collections import Counter
count = Counter(bigrams)

#print(count)

sum_ngrams = pd.DataFrame(data = list(count.values())).sum().values[0]

probability_ngrams = pd.DataFrame(data = list(count.values()),index = list(count.keys()),columns = ['probability'])/sum_ngrams


probability_ngrams.sort_values('probability',inplace=True,ascending=False)

#-- Text to Features -- Statistical Features
#%%-- Text to Features -- Statistical Features -- Tf-Idf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
#transformer   

counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]

tfidf = transformer.fit_transform(counts)
#tfidf                         



tfidf.toarray()     
#%%-- Text to Features -- Word Embedding



























