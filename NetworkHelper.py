from pybrain import structure

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

#DEFINE LAYER FUNCTIONS
def make_linear_layer(dim=1):
  return structure.LinearLayer(dim)

def make_sigmoid_layer(dim=1):
  return structure.SigmoidLayer(dim)

def make_gauss_layer(dim=1):
  return structure.GaussianLayer(dim)

def make_MDLSTM_layer(dim=1):
  return structure.MDLSTMLayer(dim)

def make_softmax_layer(dim=1):
  return structure.SoftmaxLayer(dim)

def make_tanh_layer(dim=1):
  return structure.TanhLayer(dim)

network_options = {
"feedforward" : build_feed_forward
}
