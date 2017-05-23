#! /usr/bin/python
#encoding=utf-8
import numpy as np;
import util;




class SynthTextDataFetcher():
    def __init__(self, mat_path, root_path):
        self.mat_path = mat_path
        self.root_path = root_path
        self._load_mat()
        
    @util.dec.print_calling    
    def _load_mat(self):
        data = util.io.load_mat(self.mat_path)
        self.image_paths = data['imnames'][0]
        self.image_bbox = data['wordBB'][0]
        self.txts = data['txt'][0]
        self.num_images =  len(self.image_paths)

    def get_image_path(self, idx):
        image_path = util.io.join_path(self.root_path, self.image_paths[idx][0])
        return image_path

    def get_num_words(self, idx):
        try:
            return np.shape(self.image_bbox[idx])[2]
        except: # error caused by dataset
            return 1


    def get_word_bbox(self, img_idx, word_idx):
        boxes = self.image_bbox[img_idx]
        if len(np.shape(boxes)) ==2: # error caused by dataset
            boxes = np.reshape(boxes, (2, 4, 1))
             
        xys = boxes[:,:, word_idx]
        assert(np.shape(xys) ==(2, 4))
        return np.float32(xys)
    
    def normalize_bbox(self, xys, width, height):
        xs = xys[0, :]
        ys = xys[1, :]
        
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)
        
        # bound them in the valid range
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(width, max_x)
        max_y = min(height, max_y)
        
        # check the w, h and area of the rect
        w = max_x - min_x
        h = max_y - min_y
        is_valid = True
        
        if w < 10 or h < 10:
            is_valid = False
            
        if w * h < 100:
            is_valid = False
        
        return is_valid, min_x, min_y, max_x, max_y, xys
        
    def get_txt(self, image_idx, word_idx):
        txts = self.txts[image_idx];
        clean_txts = []
        for txt in txts:
            clean_txts += txt.split()
        return str(clean_txts[word_idx])
        
        
    def fetch_record(self, image_idx):
        image_path = self.get_image_path(image_idx)
        if not (util.io.exists(image_path)):
            return None;
        img = util.img.imread(image_path)
        h, w = img.shape[0:-1];
        num_words = self.get_num_words(image_idx)
        rect_bboxes = []
        txts = []
        for word_idx in xrange(num_words):
            xys = self.get_word_bbox(image_idx, word_idx);       
            is_valid, min_x, min_y, max_x, max_y, xys = self.normalize_bbox(xys, width = w, height = h)
            if not is_valid:
                continue;
            rect_bboxes.append([min_x, min_y, max_x, max_y])
            txt = self.get_txt(image_idx, word_idx);
            txts.append(txt);
        if len(rect_bboxes) == 0:
            return None;
        
        return image_path, img, txts, rect_bboxes
    
        


def cvt_to_xmls(output_path , data_path, gt_path):

    fetcher = SynthTextDataFetcher(root_path = data_path, mat_path = gt_path)
    image_idx = -1
    while image_idx < fetcher.num_images:
        image_idx += 1;
        if image_idx >= fetcher.num_images:
            break;
        print "loading image %d/%d"%(image_idx + 1, fetcher.num_images)
        record = fetcher.fetch_record(image_idx);
        if record is None:
            print '\nimage %d does not exist'%(image_idx + 1)
            continue;
        image_path, image, txts, rect_bboxes= record;
        h, w = image.shape[0:-1]
        xmin, ymin, xmax, ymax = rect_bbox;
        labels = len(rect_bboxes) * [1];
        difficult = len(rect_bboxes) * [0];
        truncated = len(rect_bboxes) * [0];
        
        # write in xml file
        image_dir = util.io.get_dir(image_path)
        image_name = util.io.get_filename(image_path)
        xml_file = open((output_path + '/' + image_idx + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>%s</folder>\n'%(image_dir))
        xml_file.write('    <filename>' + str(image_name) + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of text on xml file
        for img_each_label in gt:
            spt = img_each_label.split(' ')
            locs = spt[0: -1]
            xmin, ymin, xmax, ymax = locs        
            xml_file.write('    <object>\n')
            xml_file.write('        <name>text</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(xmin) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(ymin) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(xmax) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(ymax) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')
                    
if __name__ == "__main__":
    mat_path = util.io.get_absolute_path('~/dataset/SynthText/gt.mat')
    root_path = util.io.get_absolute_path('~/dataset/SynthText/')
    output_dir = util.io.get_absolute_path('~/dataset/SynthText/xml')
    util.io.mkdir(output_dir);
    cvt_to_xmls(output_path = output_dir, data_path = root_path, gt_path = mat_path)






