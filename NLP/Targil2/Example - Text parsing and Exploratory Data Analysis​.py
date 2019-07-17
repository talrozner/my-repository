import pandas as pd
import numpy as np
#%% POS Tagging
from nltk.tokenize import word_tokenize
from nltk import pos_tag
sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=word_tokenize(sent)
pos_tag(tokens)


#%%-- Text to Features -- Syntactic Parsing -- Dependency Trees
#Generating Dependency Trees using Stanford Core NLP
import spacy

nlp = spacy.load('en')#'en_core_web_sm'
doc = nlp(sent)

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_)#,token.shape_, token.is_alpha, token.is_stop)
    print("\n")

#%%Named Entity Recognition
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(u"Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


#%%Named Entity Recognition
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp(u"Apple is looking at buying U.K. startup for $1 billion")
print([(X.text, X.label_) for X in doc.ents])


#%% Named Entity Recognition
from bs4 import BeautifulSoup
import requests
import re
from collections import Counter 
def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))
ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')

article = nlp(ny_bb)
len(article.ents)

labels = [x.label_ for x in article.ents]
print(Counter(labels))

items = [x.text for x in article.ents]
print(Counter(items).most_common(3))


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

print(count.most_common(3))
#print(count)

sum_ngrams = pd.DataFrame(data = list(count.values())).sum().values[0]

probability_ngrams = pd.DataFrame(data = list(count.values()),index = list(count.keys()),columns = ['probability'])/sum_ngrams


probability_ngrams.sort_values('probability',inplace=True,ascending=False)

#%% N-Grams with tags
nlp = spacy.load("en")
doc = nlp(text_ngrams)
for token in doc:
    #print (str(token.text),  str(token.lemma_),  str(token.pos_),  str(token.dep_))
    pos_sent = ["".join(str(token.pos_)) for token in doc]
    str_pos_sent = ' '.join(pos_sent)

bigrams = generate_ngrams(str_pos_sent, 2)

from collections import Counter
count = Counter(bigrams)

#print(count)

sum_ngrams = pd.DataFrame(data = list(count.values())).sum().values[0]

probability_pos_ngrams = pd.DataFrame(data = list(count.values()),index = list(count.keys()),columns = ['probability'])/sum_ngrams


probability_pos_ngrams.sort_values('probability',inplace=True,ascending=False)
#%% Relation Extraction












