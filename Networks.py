#!/usr/bin/env python

#Import PyBrain's Neural Network Structures
from pybrain import structure
import NetworkHelper as NH

class Network():
  
  specified_networks = []
  built_networks = []
  built_layers = []
  
  def __init__(self, networktype="FeedForward"):
    print "Construction Network class..."
    self.specified_networks.append(networktype)
    print "Specified Networks:", self.specified_networks

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

