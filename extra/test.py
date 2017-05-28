#! /usr/bin/env python2.7
# coding:utf-8

import os
import caffe
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import sys
import deteval
import time
import util
def get_gt(split):
    if split == 'test':
        src_img_dir = util.io.get_absolute_path("~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Test_Task12_Images")
        src_txt_dir = util.io.get_absolute_path("~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Test_Task1_GT")
    else:
        src_img_dir = util.io.get_absolute_path("~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Training_Task12_Images")
        src_txt_dir = util.io.get_absolute_path("~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Training_Task1_GT")
    
    return src_img_dir, src_txt_dir
    
def get_latest_ckpt(path):
    max_iter = 0;
    ckpt = ""
    for file in util.io.ls(path):
        if file.endswith(".caffemodel"):
          basename = os.path.splitext(file)[0]
          _iter = int(basename.split("_iter_")[1])
          if _iter > max_iter:
            ckpt = basename;
            max_iter = _iter
    return ckpt

def get_prototxt(ckpt_path):
    return "%s/deploy.prototxt"%(ckpt_path);

last_ckpt = None
def evalfixed(ckpt_path, split, confidences, gpu, postfix = None):
    global last_ckpt
    ckpt_path = util.io.get_absolute_path(ckpt_path)
    if not util.io.is_dir(ckpt_path):
        ckpt = util.io.get_filename(ckpt_path)
        startPosition = ckpt.rfind('/') + 1
        endPosition = ckpt.find('.caffemodel')
        ckpt_name = ckpt[startPosition:endPosition]
        ckpt_path = util.io.get_dir(ckpt_path)
    else:
        ckpt_name = get_latest_ckpt(ckpt_path)
    
    if last_ckpt == None or ckpt_name != last_ckpt:
        last_ckpt = ckpt_name
    else:
        return
        
    model_prototxt = get_prototxt(ckpt_path)
    model_weights = "%s/%s.caffemodel"%(ckpt_path, ckpt_name)
    if not util.io.exists(model_weights):
        return
    labelmap_file = "data/icdar2013/labelmap.prototxt"
    im_dir, gt_dir = get_gt(split)
    result_path = util.io.join_path(ckpt_path, 'result')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if gpu > 0:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    file_label = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file_label.read()), labelmap)

    net = caffe.Net(model_prototxt, model_weights, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    image_resize = 512
    net.blobs['data'].reshape(1, 3, image_resize, image_resize)


        
    image_names = os.listdir(im_dir)
    dump_paths = []
    zip_paths = []
    txt_dump_paths = []
    xml_dump_paths = []
    eval_result_paths = []
    for img_idx, image_name in enumerate(image_names):
        if not image_name.endswith('.jpg'):
            continue
        image_path = os.path.join(im_dir, image_name)
        image = caffe.io.load_image(image_path)
        print "%d/%d:"%(img_idx, len(image_names)), image_name

        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        detections = net.forward()['detection_out']

        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]
        for confidence in confidences:
            dump_path = util.io.join_path(result_path, ckpt_name , 'confidence_' + str(confidence))
            if postfix is not None:
                dump_path += postfix
            img_dump_path = util.io.join_path(dump_path, 'vis')
            txt_dump_path = util.io.join_path(dump_path, 'txt')
            xml_dump_path = util.io.join_path(dump_path, 'xml')
            eval_result_path = util.io.join_path(dump_path, 'evalfixed.xml')
            zip_path = util.io.join_path(dump_path, "%s_%s_confidence=%f.zip"%(ckpt_name, split, confidence))
            util.io.mkdir(img_dump_path)
            util.io.mkdir(txt_dump_path)
            if img_idx == 0:
                dump_paths.append(dump_path)
                zip_paths.append(zip_path) 
                txt_dump_paths.append(txt_dump_path)
                xml_dump_paths.append(xml_dump_path)
                eval_result_paths.append(eval_result_path)

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= confidence]

            top_conf = det_conf[top_indices]
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            rects = []
            
            image = util.img.imread(image_path)
            for i in xrange(top_conf.shape[0]):
                score = top_conf[i]
                xmin = int(round(top_xmin[i] * image.shape[1]))
                ymin = int(round(top_ymin[i] * image.shape[0]))
                xmax = int(round(top_xmax[i] * image.shape[1]))
                ymax = int(round(top_ymax[i] * image.shape[0]))
                util.img.rectangle(image, (xmin, ymin), (xmax, ymax), color = util.img.COLOR_GREEN, border_width = 2)
                util.img.put_text(image, text = str(score), pos = (xmin, ymin), scale = 0.5, color = util.img.COLOR_WHITE, thickness = 1);
                util.img.imwrite(util.io.join_path(img_dump_path, image_name + '.jpg'), image)
                rects.append('%s, %s, %s, %s' % (xmin, ymin, xmax, ymax))
            

            dst_path = util.io.get_absolute_path(util.io.join_path(txt_dump_path, image_name[:image_name.find('.jpg')] + '.txt'))
            with open(dst_path, 'w') as f:
                f.writelines("%s\n" % str(item) for item in rects)

    # deteval of all results                
    for dump_path, zip_path, txt_dump_path, xml_dump_path, eval_result_path in zip(dump_paths, zip_paths, txt_dump_paths, xml_dump_paths, eval_result_paths):
        # create zip file
        cmd = 'cd %s;zip -j %s %s/*'%(dump_path, zip_path, txt_dump_path);
        print cmd
        util.cmd.cmd(cmd);
        print "zip file created: ", util.io.join_path(dump_path, zip_path)
        
        # evalfixed
        deteval.eval(det_txt_dir = txt_dump_path, gt_txt_dir = gt_dir, xml_path = xml_dump_path, write_path = eval_result_path);        

if __name__ == "__main__":
    
    ckpt_path = sys.argv[1]
    split = sys.argv[2]
    gpu = int(sys.argv[3])
    wait = bool(int(sys.argv[4]))
    confidences = [0.05, 0.2, 0.25, 0.3, 0.4, 0.5, 0.1, 0.15, 0.125]
    #confidences = [0.125]
    while True: 
        evalfixed(ckpt_path, split, confidences=confidences, gpu = gpu, postfix = '_nms_threshold_0.45')
        if wait:
            print "waiting for new checkpoint..."
            time.sleep(60)
        else:
            break;
