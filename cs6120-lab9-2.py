from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('vectors-words.bin', binary=True)

vector = model['king']
print(vector)