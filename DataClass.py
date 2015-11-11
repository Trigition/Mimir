#!/usr/bin/env python

import abc
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

  def __init__(self, data, description=None):
    self.data_unit = data
    self.description = description

  @abc.abstractmethod
  def print_data():
    """This method should be overriden to print out the data contained
    within the class"""
    return
    
class ImageData(Data):
  
  def print_data():
    print "Image Object:" + str(this.data_unit)

  def add_image():

class ClassificationSet():
  in_dimension = 1
  out_dimension = 1
  n_classes = 1
  class_data_set = None

  def __init__(self, in_dimension, out_dimension, n_classes):
    self.in_dimension = in_dimension
    self.out_dimension = out_dimension
    self.n_classes = n_classes

  def init_class_data_set(self):
    self.class_data_set = ClassificationDataSet(in_dimension, out_dimension, n_classes)
