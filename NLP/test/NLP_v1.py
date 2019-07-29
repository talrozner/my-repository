#NLP - V1
#%% -- information extraction

#If this location data was stored in Python as a list of tuples (entity, relation, entity), then the question "Which organizations operate in Atlanta?" could be translated as follows:

locs = [('Omnicom', 'IN', 'New York'),
         ('DDB Needham', 'IN', 'New York'),
         ('Kaplan Thaler Group', 'IN', 'New York'),
         ('BBDO South', 'IN', 'Atlanta'),
         ('Georgia-Pacific', 'IN', 'Atlanta')]

query = [e1 for (e1, rel, e2) in locs if e2=='Atlanta']
print(query)
 
 
#%%-- Information Extraction Architecture
import nltk
def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

document = 'The Lion King is a 1994 American animated musical film produced by Walt Disney Feature Animation and released by Walt Disney Pictures. It is the 32nd Disney animated feature film, and the fifth animated film produced during a period known as the Disney Renaissance. The Lion King was directed by Roger Allers and Rob Minkoff, produced by Don Hahn, and has a screenplay credited to Irene Mecchi, Jonathan Roberts, and Linda Woolverton. Its original songs were written by composer Elton John and lyricist Tim Rice, with a score by Hans Zimmer. The film features an ensemble voice cast that includes Matthew Broderick, James Earl Jones, Jeremy Irons, Jonathan Taylor Thomas, Moira Kelly, Nathan Lane, Ernie Sabella, Rowan Atkinson, Robert Guillaume, Madge Sinclair, Whoopi Goldberg, Cheech Marin, and Jim Cummings. The story takes place in a kingdom of lions in Africa and was influenced by William Shakespeares Hamlet.'


ie_document = ie_preprocess(document)

#--Chunking
#%%--Chunking -- Noun Phrase Chunking
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]#ie_document[0] ## take the first sentence

grammar = "NP: {<DT>?<JJ>*<NN>}"

cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)
result.draw() 
#%%--Chunking -- Tag Patterns

#%%--Chunking -- Chunking with Regular Expressions
	
grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
cp = nltk.RegexpParser(grammar)
sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
print(cp.parse(sentence))


 	
nouns = [("money", "NN"), ("market", "NN"), ("fund", "NN")]
grammar = "NP: {<NN><NN>}  # Chunk two consecutive nouns"
cp = nltk.RegexpParser(grammar)
print(cp.parse(nouns))


#%%--Named Entity Recognition
sent = nltk.corpus.treebank.tagged_sents()
print(nltk.ne_chunk(sent[0], binary=True))



#%%--Relation Extraction
import re
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
     for rel in nltk.sem.extract_rels('ORG', 'LOC', doc,corpus='ieer', pattern = IN):
         print(nltk.sem.rtuple(rel))





























