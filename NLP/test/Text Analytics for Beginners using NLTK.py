#%% import
p_num = 1
# Text Analysis Operations using NLTK
#%% Sentence Tokenization
from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)

#%% Word Tokenization
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

#%% Frequency Distribution
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)

fdist.most_common(2)

#%% Frequency Distribution Plot
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.xlabel('word', fontsize=50)
plt.ylabel('frequency', fontsize=50)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.show()
p_num+=1

#%% Stopwords
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)

#%%Removing Stopwords
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 

def clean_text(text):
    text = text.lower()
    text = [word for word in text.split(' ') if not word in stop_words]
    text = " ".join(text)
    return text

new_text = clean_text(text)

print("clean text: " +"\n" + str(new_text))

#Lexicon Normalization
#%% Stemming


















