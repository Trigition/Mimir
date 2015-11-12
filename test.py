#!/usr/bin/env python

print "Importing libraries..."
from Networks import Network
from DataClass import ImageData
from sklearn import datasets

print "Obtaining datasets..."
olivetti = datasets.fetch_olivetti_faces()

Images, Targets = olivetti.data, olivetti.target

dim = Images.shape
print "DIMENSION:", dim
output = open("test.out", "w")
output.write("Epoch,Error\n")
for i in range(1000):  
  imageData = ImageData(Images, Targets, 64, 64, outputs = 1)
  imageData.train_all_data(0.25)

  testNet = Network(imageData, 64, "Softmax")
  testNet.init_backprop_trainer(b_verbose=False)
  error = testNet.run_network(i)
  print "Epochs:", i,"Error:", error
  output.write(str(i) + "," + str(error) + "\n")
