import caffe
import numpy as np

def cal(v):
	return np.sqrt(np.sum(v*v))
class DebugLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass
        
    def reshape(self, bottom, top):
	    pass

    def forward(self, bottom, top):
        import pdb
        pdb.set_trace()        
        data = bottom[0].data[...]
        v = data[0, :, 0, 0]
        print cal(v) 
        data = bottom[1].data[...]
        v = data[0, :, 0, 0]
        print cal(v) 
	
    def backward(self, top, propagate_down, bottom):
        pass
