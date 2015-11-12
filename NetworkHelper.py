from pybrain.structure.modules import LinearLayer
from pybrain.structure.modules import SigmoidLayer
from pybrain.structure.modules import GaussianLayer
from pybrain.structure.modules import MDLSTMLayer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import TanhLayer

def build_feed_forward():
  built_layers = []
  built_layers.append(make_linear_layer(2))
  built_layers.append(make_sigmoid_layer(3))
  built_layers.append(make_linear_layer(1))
  #DoThings
  network = structure.FeedForwardNetwork()
  in_layer = built_layers[0]
  hidden_layer = built_layers[1]
  out_layer = built_layers[2]
  network.addInputModule(in_layer)
  network.addModule(hidden_layer)
  network.addOutputModule(out_layer)

  to_hidden = structure.FullConnection(in_layer, hidden_layer)
  to_out = structure.FullConnection(hidden_layer, out_layer)

  network.addConnection(to_hidden)
  network.addConnection(to_out)

  network.sortModules()
  return network

def build_recurrent():
  print "IMPLEMENT!"   

#DEFINE LAYER FUNCTIONS
layers = {
  "Linear" : LinearLayer,
  "Sigmoid" : SigmoidLayer,
  "Guassian" : GaussianLayer,
  "MDLSTML" : MDLSTMLayer,
  "Softmax" : SoftmaxLayer,
  "Tanh" : TanhLayer
  }
network_options = {
"feedforward" : build_feed_forward
}

