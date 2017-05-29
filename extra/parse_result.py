import util
import sys
import glob

result_dir = util.io.get_absolute_path(sys.argv[1])
xmls = glob.glob(r'%s/*/*/evalfixed.xml'%(result_dir))
def get_iter(xml):
    data = util.str.find_all(xml, '_iter_\d+')[0]
    iter = int(data.split('_')[-1])
    return iter
def get_neg_overlap(xml):
    data = util.str.find_all(xml, 'neg_overlap_0\.\d+')[0]
    overlap = float(data.split('_')[-1])
    return overlap
def get_confidence(xml):
    data = util.str.find_all(xml, 'confidence_0\.\d+')[0]
    confidence = float(data.split('_')[-1])
    return confidence
def get_nms_threshold(xml):
    data = util.str.find_all(xml, 'nms_threshold_0\.\d+')[0]
    nms_threshold = float(data.split('_')[-1])
    return nms_threshold

def get_result_str(xml):
    content = util.io.cat(xml)
    data = util.str.find_all(content, '\<score.+\/\>')[0]
    return data
class Result(object):
    def __init__(self, xml):
        self.iteration = get_iter(xml)
        self.neg_overlap = get_neg_overlap(xml)
        self.confidence = get_confidence(xml)
        self.nms_threshold = get_nms_threshold(xml)
        self.data = get_result_str(xml)

    def Print(self):
        print "|%d|%f|%f|%f|`%s`|"%(self.iteration, self.neg_overlap, self.confidence, self.nms_threshold, self.data)
def sort_by_confidence(r1, r2):
    if r1.iteration == r2.iteration:
        return -1 if r1.confidence < r2.confidence else 1
    else:
        return r1.iteration - r2.iteration
results = []
for xml in xmls:
    result = Result(xml)
    results.append(result)

results.sort(sort_by_confidence)

print "|Iter|neg_overlap|confidence|nms_threshold|result|"
print "|---|---|---|---|---|"
for r in results:
    r.Print()
