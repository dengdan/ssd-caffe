#! /usr/bin/env python2.7
# coding:utf-8

import os
import caffe
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def detection_test_set(model_def,
                       model_weights,
                       image_dir,
                       resultPath,
                       labelmap_file,
                       thrs=[0.5],
                       mode_CPU_GPU='GPU'):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if mode_CPU_GPU == 'GPU':
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    file_label = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file_label.read()), labelmap)

    net = caffe.Net(model_def, model_weights, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    image_resize = 512
    net.blobs['data'].reshape(1, 3, image_resize, image_resize)

    for image_name in os.listdir(image_dir):
        if not image_name.endswith('.jpg'):
            continue
        image_path = os.path.join(image_dir, image_name)
        image = caffe.io.load_image(image_path)
        print image_name

        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        detections = net.forward()['detection_out']

        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        for thr in thrs:
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= thr]

            top_conf = det_conf[top_indices]
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            rects = []
            startPosition = model_weights.rfind('/') + 1
            endPosition = model_weights.find('.caffemodel')
            dump_path = util.io.join_path(resultPath, model_weights[startPosition:endPosition] , 'thr_' + str(thr))
            img_dump_path = util.io.join_path(dump_path, 'vis')
            txt_dump_path = util.io.join_path(dump_path, model_weights[startPosition:endPosition] + '_thr_' + str(thr))
            util.io.mkdir(img_dump_path)
            util.io.mkdir(txt_dump_path)
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
                rects.append('%s, %s, %s, %s' % (xmin, ymin, xmax-xmin+1, ymax-ymin+1))

            dst_path = util.io.get_absolute_path(util.io.join_path(txt_dump_path, image_name[:image_name.find('.jpg')] + '.txt'))
            with open(dst_path, 'w') as f:
                f.writelines("%s\n" % str(item) for item in rects)


if __name__ == "__main__":
    import util
#    model_prototxt = util.io.get_absolute_path("~/temp/no-use/deploy.prototxt")
    model_prototxt = "models/VGGNet/jm/SSD_512x512/deploy.prototxt"
    model_weights = "models/VGGNet/jm/SSD_512x512/VGG_jm_SSD_512x512_iter_20000.caffemodel"
    im_dir = util.io.get_absolute_path("~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Test_Task12_Images")

    result_path = "~/temp/no-use/caffe-result/jm2"
    labelmap_file = "data/icdar2013/labelmap.prototxt"
    detection_test_set(model_prototxt, model_weights, im_dir, result_path, labelmap_file, thrs=[0.15, 0.2, 0.25, 0.5])

