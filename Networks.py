#!/usr/bin/env python

#Import PyBrain's Neural Network Structures
from pybrain import structure
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer

from DataClass import Data

import NetworkHelper as NH

class Network():
  
  network = None
  trainer = None
  hidden_layer = None
  hidden_nodes = None
  data = None

  def __init__(self, data, n_hidden_nodes, layertype="Linear"):
    self.hidden_layer = NH.layers[layertype]
    self.data = data

    train_dim_in = data.train_data.indim
    train_dim_out = data.train_data.outdim

    self.network = buildNetwork(train_dim_in, n_hidden_nodes, train_dim_out, outclass=self.hidden_layer)
    self.hidden_nodes = n_hidden_nodes

  def init_backprop_trainer(self, b_momentum=0.1, b_learningrate=0.01, b_verbose=True, b_weightdecay=0.1):
    train_in = self.data.train_data.indim
    train_out = self.data.train_data.outdim
    self.trainer = BackpropTrainer(self.network, dataset=self.data.train_data, \
                                  momentum=b_momentum, learningrate=b_learningrate, verbose=b_verbose, \
                                  weightdecay=b_weightdecay)
  
  def run_network(self, epoch):
    NetworkWriter.writeToFile(self.network, "test.xml")
    self.trainer.trainEpochs(epoch)
    error = percentError(self.trainer.testOnClassData(dataset=self.data.test_data), \
          self.data.test_data['class'])
    return error

  
  """
  def build_networks(self):
    for specification in self.specified_networks:
      specification = specification.lower()
      print "Attempting to build:", specification
      try:
        self.built_networks.append(NH.network_options[specification]())
      except KeyError:
        print "ERROR:", specification, "is undefined!"
    print "Built Networks..."
    for network in self.built_networks:
      print "NETWORK", network
"""
