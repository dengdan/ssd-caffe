import caffe
import numpy as np
class DebugLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
        
    def reshape(self, bottom, top):
	    pass

    def forward(self, bottom, top):
        import pdb
        pdb.set_trace()        
        data = bottom[0].data[...]
        label = bottom[1].data[...]
        v = data[0, :, 0, 0]
        print cal(v) 
        print label
	
    def backward(self, top, propagate_down, bottom):
        pass
