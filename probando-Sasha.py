# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:55:06 2018

@author: ssbs3
"""

import numpy as np  # a conventional alias
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as LSA
from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

filenames = ['./data/laDuda.txt','./data/eljorge.txt','./data/llecspier.txt','./data/vairon.txt']

vectorizer = CountVectorizer(input='filename',ngram_range = (1,3), stop_words = "english")

dtm = vectorizer.fit_transform(filenames)  # a sparse matrix

tfidf_transformer = TfidfTransformer(norm = 'l2')
x_tfidf = tfidf_transformer.fit_transform(dtm)
matVoc = x_tfidf.toarray()

vocab = vectorizer.get_feature_names()  # a list
vocab = np.array(vocab)