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
  def train_all_data(self, train_data_ratio, expand=False):
      self.test_data, self.train_data = self.classifier.splitWithProportion(train_data_ratio)
      if expand:
          self.test_data._convertToOneOfMany()
          self.train_data._convertToOneOfMany()

  def get_test_train(self):
      return (self.test_data, self.train_data)

class ImageData(Data):
  
  image_x = 1
  image_y = 1
  images = []
  targets = []

  def __init__(self, description="Image Data", images, targets, image_x, image_y, outputs=1):
      Data.__init__(self, description, outputs)
      self.images = images
      self.targets = targets
      self.images_x = image_x
      self.images_y = image_y

  def create_classifier(self):
      vector_length = self.image_x * self.image_y
      #Create the classifier
      self.classifier = ClassificationDataSet(vector_length, self.outputs, len(images))
      for i in xrange(len(images)):
          #Assign images to their targets in the classifier
          self.classifier.addSample(np.ravel(self.images[i]), targets[i])

  def print_data(self):
    print "Image Object:" + str(this.data_unit)
    
  def add_image(self, image, target):
    self.images.append(image)
    self.targets.append(target)
