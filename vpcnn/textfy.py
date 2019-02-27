import gensim
from gensim.models.keyedvectors import KeyedVectors

w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
text = open('w2v.300d.txt', 'w')

for word in w2v.__dict__['vocab']:
	arr = w2v[word]
	#print(arr)
	string = ' '.join([word, ' '.join([str(x) for x in arr])])
	print(string, file=text)
