import bs4 as bs  
import urllib.request  
import re
import nltk
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
#function that give the part of speech!!!
def return_parts_of_speech(word):
    nlp = spacy.load("en")
    doc = nlp(word)
    for token in doc:
        parts_of_speech = (str(token.lemma_),  str(token.pos_),str(token.tag_) , str(token.dep_))
    return parts_of_speech
        
#function for NER!!!
def return_is_GPE(sent):
    is_GPE = False
    nlp = en_core_web_sm.load()
    doc = nlp(sent)
    #print([(X.text, X.label_) for X in doc.ents])    
    entity_list = [(X.text, X.label_) for X in doc.ents]
    for i in entity_list:
        if i[1] == 'GPE':
            is_GPE = True
            break
    return is_GPE

# convert text to pos!!!
def convert_text_to_pos(text):
    nlp = spacy.load("en")
    doc = nlp(text)
    for token in doc:
        #print (str(token.text),  str(token.lemma_),  str(token.pos_),  str(token.dep_))
        pos_sent = ["".join(str(token.pos_)) for token in doc]
        str_pos_sent = ' '.join(pos_sent)
    return str_pos_sent

#create n-grames
def generate_ngrams(text, n):
    words = text.split()
    output = []  
    for i in range(len(words)-n+1):
        output.append(tuple(words[i:i+n]))
    return output

#Fetching Articles from Wikipedia
scraped_data = urllib.request.urlopen('http://lite.cnn.io/en/article/h_0a195a79d692ffc3c4eeeefc6db2e825')  ##'https://en.wikipedia.org/wiki/Artificial_intelligence'#'https://en.wikipedia.org/wiki/House_of_Medici'
article = scraped_data.read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:  
    article_text += p.text

#Preprocessing

# Removing Square Brackets and Extra Spaces
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
article_text = re.sub(r'\s+', ' ', article_text)  

# Removing special characters and digits
formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)  



#convert text to pos!!!
formatted_article_pos = convert_text_to_pos(formatted_article_text)
bigrams = generate_ngrams(formatted_article_pos, 2)
from collections import Counter
count = Counter(bigrams)
most_common_pos = count.most_common(1)[0][0]



#Converting Text To Sentences
sentence_list = nltk.sent_tokenize(article_text)

#Find Weighted Frequency of Occurrence
stopwords = nltk.corpus.stopwords.words('english')

word_frequencies = {}  
for word in nltk.word_tokenize(formatted_article_text):  
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

maximum_frequncy = max(word_frequencies.values())

for word in word_frequencies.keys():  
    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
 
p = 0
#Change Word Score!!!
for word in word_frequencies.keys():  
    p+=1
    word_lemma , word_pos , word_tag , word_dep = return_parts_of_speech(word)
    #print(len(word_frequencies.keys())-p)
    #print(word_lemma)
    if word_lemma == 'be':
        word_frequencies[word] = word_frequencies[word] * 0.8
        print('be')
    #elif p >=3:
     #   break
    elif word_pos == 'PROPN':
        word_frequencies[word] = word_frequencies[word] * 1.3
        print('PROPN')
    elif word_tag == 'IN':
        word_frequencies[word] = word_frequencies[word] * 0.5
        print('IN')
    elif word_dep == 'pcomp':
        word_frequencies[word] = word_frequencies[word] * 0.9
        print('pcomp')
    
p = 0
#Calculating Sentence Scores
sentence_scores = {}  
for sent in sentence_list:  
    p+=1
    print(len(sentence_list)-p)
    #if 'GPE' entity in sent then put high score!!!
    if return_is_GPE(sent) == True:
        print("GPE")
        sentence_scores[sent] = 1000
        continue
    
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

import heapq  
summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)  
print(summary)  

























