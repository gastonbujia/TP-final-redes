# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:32:32 2017

@author: Fede albanese

"""




import numpy as np  # a conventional alias
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as LSA
from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


filenames = ['laDuda.txt','eljorge.txt','llecspier.txt','vairon.txt']

vectorizer = CountVectorizer(input='filename',ngram_range = (1,3), stop_words = "english")

dtm = vectorizer.fit_transform(filenames)  # a sparse matrix

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(dtm)
matVoc = x_tfidf.toarray()


vocab = vectorizer.get_feature_names()  # a list
vocab = np.array(vocab)

lda = LDA(n_components=3,   random_state=0)

lda_array = lda.fit_transform(matVoc)

labels = [np.argmax(x) for x in lda_array]

print(lda_array)

colores = ["r","b","c","y"]
autores = ["Poema","Chapman","Shakespeare","Lord Byron"]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i,punto in enumerate(lda_array):
    ax.scatter(punto[0],punto[1],punto[2], color = colores[i],label=autores[i])

ax.legend()
ax.set_xlim(0,0.15)
ax.set_ylim(0.8,1)
ax.set_zlim(0,0.15)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()





#calcular distancias
print("Distancias al poema: ")
for punto,texto in zip(lda_array,autores):
    print(texto + ": ",np.sqrt(sum([(i-j)**2 for i,j in zip(punto,lda_array[0])])))

