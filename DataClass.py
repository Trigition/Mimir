#!/usr/bin/env python

import abc
import numpy as np
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

class Data:
  __metaclass__ = abc.ABCMeta
  data_unit = None
  description = None
  classifier = None
  outputs = 1

  test_data = None
  train_data = None

  def __init__(self, data, description=None, outputs=1):
    self.data_unit = data
    self.description = description
    self.outputs = outputs

  @abc.abstractmethod
  def print_data(self):
    """This method should be overriden to print out the data contained
    within the class"""
    return
  
  #Returns with Test Data, Train Data
  def train_all_data(self, train_data_ratio):
      self.test_data, self.train_data = self.classifier.splitWithProportion(train_data_ratio)
      self.test_data._convertToOneOfMany()
      self.train_data._convertToOneOfMany()

  def get_test_train(self):
      return (self.test_data, self.train_data)

class ImageData(Data):
  
  image_x = 1
  image_y = 1
  images = []
  targets = []

  def __init__(self, images, targets, image_x, image_y, description="Image Data", outputs=1):
      Data.__init__(self, description, outputs)
      self.images = images
      self.targets = targets
      self.image_x = image_x
      self.image_y = image_y
      self.create_classifier()

  def create_classifier(self):
      #print "Image X:", self.image_x
      #print "Image Y:", self.image_y
      vector_length = self.image_x * self.image_y
      #Create the classifier
      #print "Creating Classifier. Vector_Len:", vector_length, "Output Vector:", self.outputs
      self.classifier = ClassificationDataSet(vector_length, self.outputs, nb_classes=(len(self.images) / 10))
      #print "Adding samples for", len(self.images), " images"
      for i in xrange(len(self.images)):
          #Assign images to their targets in the classifier
          #print i, "Image:", self.images[i], "Target:", self.targets[i]
          self.classifier.addSample(self.images[i], self.targets[i])

  def print_data(self):
    print "Image Object:" + str(this.data_unit)
    
  def add_image(self, image, target):
    self.images.append(image)
    self.targets.append(target)
