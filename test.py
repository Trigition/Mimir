#!/usr/bin/env python

print "Importing libraries..."
from Networks import Network
from DataClass import ImageData
from sklearn import datasets

print "Obtaining datasets..."
olivetti = datasets.fetch_olivetti_faces()

Images, Targets = olivetti.data, olivetti.target

dim = Images.shape
imageData = ImageData(Images, Targets, dim[0], dim[0], outputs = 1)
imageData.train_all_data(0.25, expand=True)

trn_data, test_data = imageData.get_test_train()
print trn_data['input'], trn_data['target'], test_data.indim, test_data.outdim

print "Starting test:"
testNetwork = Network("FeedForward")
testNetwork.build_networks()
