import os
import caffe
import numpy as np
import sys
import logging
import setproctitle
setproctitle.setproctitle('caffe-ssd-text')


'''
layer {
  name: 'debug1'
  type: 'Python'  
  bottom: 'conv4_3'
  bottom: 'conv4_3_norm'
  top: 'd'
  python_param {
    module: 'debug_layer'
    layer: 'DebugLayer'  
  }
}
'''

import util
util.mod.add_to_path('extra')
model_path = 'models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel'
model_path = util.io.get_absolute_path(model_path)

# init
caffe.set_mode_gpu()
caffe.set_device(int(sys.argv[1]))
logging.info('using GPU %d'%(int(sys.argv[1])))
print util.io.cat('models/VGGNet/icdar2013/SSD_512x512/solver.prototxt')
solver = caffe.SGDSolver('models/VGGNet/icdar2013/SSD_512x512/solver.prototxt')
solver.net.copy_from(model_path)

for iteration in xrange(1000000):
    solver.step(1)
    

