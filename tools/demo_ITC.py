#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import glob

CLASSES = ('__background__',
           'hand')

NETS = {'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_70000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def vis_detections_opencv(im, class_name, dets, thresh, color):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im 
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(im, "%s %.2f" % (class_name, score), (bbox[0], int(bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.cv.CV_AA)   
    return im

def _write_inria_results_file(all_boxes, path):
    for cls_ind, cls in enumerate(CLASSES):
        if cls == '__background__':
            continue
        filename = path + 'det_demo_' + cls + '.txt'
        print 'Writing %s results file into location %s' % (cls, filename)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # I have modifed the following code to match VIVA standard
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f} {:d} {:d} {:d}\n'.
                            format(index, 
                                   dets[k, 0], dets[k, 1],
                                   dets[k, 2] - dets[k, 0], dets[k, 3] - dets[k,1],
                                   dets[k, -1], -1, -1, -1))
        
def demo(net, im_file, path, folderName, CONF_THRESH):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'itc/preeti/20151030_150729/hand', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        inds = np.where(scores[:, cls_ind] >= CONF_THRESH)[0]
        cls_boxes = boxes[inds, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[inds, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        im = vis_detections_opencv(im, cls, dets, thresh=CONF_THRESH, color = (0,0,255))

        filename = 'ITC_dets/'+folderName+'.txt'
        with open(filename, 'a') as f:
            if dets == []:
                continue
            # I have modifed the following code to match VIVA standard
            #print "Starts writing into file!!!!!!"
            for k in xrange(dets.shape[0]):
                f.write('{:s} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f} {:d} {:d} {:d}\n'.
                        format(im_file[im_file.rfind('/')+1:im_file.rfind('.jpg')],
                               dets[k, 0], dets[k, 1],
                               dets[k, 2] - dets[k, 0], dets[k, 3] - dets[k,1],
                               dets[k, -1], -1, -1, -1))

    return im

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--folderName', dest='folderName', help='the folder name for the drive',
                        default='1458587118485511')
    parser.add_argument('--handFolder', dest='handFolder', help='the folder name for hand',
                        default='handcam')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'tian_faster_rcnn_models',
                              NETS[args.demo_net][1])
    print "prototxt: ", prototxt
    print "caffemodel: ", caffemodel
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    print '\n\nStarts loading network {:s}'.format(caffemodel)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _= im_detect(net, im)

    # Init video write
    out = cv2.VideoWriter(filename = 'ITC_dets/'+args.folderName +'_'+ args.handFolder +'.avi', fourcc = cv2.cv.CV_FOURCC(*'XVID'), fps = 10, frameSize = (640, 480))
    path = '/mnt/disk3/fy15_sakiyomiagent_datacollection/polysync/polysync_processed/'+args.folderName+'/'
    imFolder = path + args.handFolder + '/*.jpg'

    print "Search in folder ", imFolder
    im_names = glob.glob(imFolder)
    im_names = sorted(im_names)

    #im_names = im_names[1000:]
    print "%i images found in folder. " % len(im_names)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {0}'.format(im_name)
        img = demo(net, im_name, path, folderName = args.folderName, CONF_THRESH=0.8)     
    	cv2.imshow('img', img)
        key = cv2.waitKey(1)
        if key == 27:
        	break
    	img2 = cv2.resize(img, (640, 480), interpolation = cv2.INTER_CUBIC)
    	out.write(img2)
    out.release()
