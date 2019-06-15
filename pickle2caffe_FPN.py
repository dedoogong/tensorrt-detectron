# encoding: utf-8
"""
@author: Seunghyun Lee
@contact: dedoogong@gmail.com
"""
from __future__ import print_function, division
import sys
import numpy as np
import pickle
import os
import pickle

os.environ['GLOG_minloglevel'] = '2'
HOME_PATH = '/home/ktai01'
caffe_root = HOME_PATH + '/caffe_vistool/caffe/'
sys.path.insert(0, caffe_root)
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, caffe_root + 'lib')
sys.path.insert(0, caffe_root + 'lib/data_provider')
sys.path.insert(0, caffe_root + 'lib/rpn')
import caffe

# CAUTION:
# 1. MUST USE FUSED GENERALIZED_CNN(part1).pkl! PART2/KEY/MASK RCNN pkl has wrong(initialized) params!!
# 2. IF IT HAS 'blobs' in its dictionaory, it means that I MUST FUSE THE MODEL!!!
caffe2kptmodel = {}
with open(HOME_PATH + '/detectron/tools/KRCNN_FPN_R50_RoiAlign.pkl', 'rb') as handle:  # model_final
    caffe2kptmodel.update(pickle.load(handle))
# caffe2kptmodel= caffe2kptmodel_['blobs']
caffe2kptmodel_keys = caffe2kptmodel.keys()
caffe2kptmodel_keys.sort()
conv1_w = caffe2kptmodel['conv1_w']
caffe_net = caffe.Net(HOME_PATH + '/detectron/tools/keypoint_rcnn_R-50-FPN-body.prototxt', caffe.TEST)
for caffe_net_key in caffe_net.params.keys():
    print(caffe_net_key)
print('--------------------------------------------------')

# BACKBONE
# print(caffe_net.params['res3_1_branch2b'][0].data[...])
caffe_net.params['conv1'][0].data[...] = caffe2kptmodel['conv1_w']  # conv1_b exist!~! hum?
# caffe_net.params['conv1'][1].data[...] = caffe2kptmodel['conv1_b']
caffe_net.params['conv1'][1].data[...] = caffe2kptmodel['res_conv1_bn_b']

def biasOrWeightWithSpecificLayer(caffe2kptmodel_key, caffe_net, specificLayer, caffe2kptmodel):
    if caffe2kptmodel_key.endswith('_b'):
        caffe_blob_name = caffe2kptmodel_key[:-2]
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][1].data[...] = caffe2kptmodel[specificLayer + '_b']
        caffe_blob_name = caffe_blob_name.replace('2', '3')
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][1].data[...] = caffe2kptmodel[specificLayer + '_b']
        caffe_blob_name = caffe_blob_name.replace('3', '4')
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][1].data[...] = caffe2kptmodel[specificLayer + '_b']
        caffe_blob_name = caffe_blob_name.replace('4', '5')
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][1].data[...] = caffe2kptmodel[specificLayer + '_b']
        caffe_blob_name = caffe_blob_name.replace('5', '6')
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][1].data[...] = caffe2kptmodel[specificLayer + '_b']
    elif caffe2kptmodel_key.endswith('_w'):
        caffe_blob_name = caffe2kptmodel_key[:-2]
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][0].data[...] = caffe2kptmodel[specificLayer + '_w']
        caffe_blob_name = caffe_blob_name.replace('2', '3')
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][0].data[...] = caffe2kptmodel[specificLayer + '_w']
        caffe_blob_name = caffe_blob_name.replace('3', '4')
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][0].data[...] = caffe2kptmodel[specificLayer + '_w']
        caffe_blob_name = caffe_blob_name.replace('4', '5')
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][0].data[...] = caffe2kptmodel[specificLayer + '_w']
        caffe_blob_name = caffe_blob_name.replace('5', '6')
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][0].data[...] = caffe2kptmodel[specificLayer + '_w']

def biasOrWeight(caffe2kptmodel_key, caffe_net, caffe2kptmodel):
    if caffe2kptmodel_key.endswith('_b'):
        caffe_blob_name = caffe2kptmodel_key.replace('_b', '')
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][1].data[...] = caffe2kptmodel[caffe2kptmodel_key]
    elif caffe2kptmodel_key.endswith('_w'):
        caffe_blob_name = caffe2kptmodel_key.replace('_w', '')
        print(caffe_blob_name)
        caffe_net.params[caffe_blob_name][0].data[...] = caffe2kptmodel[caffe2kptmodel_key]

for caffe2kptmodel_key in caffe2kptmodel_keys:
    if not 'fcn' in caffe2kptmodel_key and \
            not 'pose' in caffe2kptmodel_key and \
            not 'fc1000' in caffe2kptmodel_key and \
            not 'kps' in caffe2kptmodel_key and \
            not 'res_conv' in caffe2kptmodel_key and \
            not 'anchor' in caffe2kptmodel_key:

        if 'res' in caffe2kptmodel_key:
            if caffe2kptmodel_key.endswith('_bn_b'):  # resnet branch's bias
                if not '_bn_' in caffe2kptmodel_key:
                    caffe_blob_name = caffe2kptmodel_key.replace('_b', '')
                    caffe_blob_name = caffe_blob_name.replace('_0', 'a').replace('_1', 'b').replace('_2','c').replace(
                        '_3', 'd').replace('_4', 'e').replace('_5', 'f')
                    print(caffe_blob_name)
                    caffe_net.params[caffe_blob_name][1].data[...] = caffe2kptmodel[caffe2kptmodel_key]
                else:
                    caffe_blob_name = caffe2kptmodel_key.replace('_bn_b', '')
                    caffe_blob_name = caffe_blob_name.replace('_0', 'a').replace('_1', 'b').replace('_2','c').replace(
                        '_3', 'd').replace('_4', 'e').replace('_5', 'f')
                    print(caffe_blob_name)
                    caffe_net.params[caffe_blob_name][1].data[...] = caffe2kptmodel[caffe2kptmodel_key]

            elif caffe2kptmodel_key.endswith('_w'):  # resnet branch's weight
                caffe_blob_name = caffe2kptmodel_key.replace('_w', '')
                caffe_blob_name = caffe_blob_name.replace('_0', 'a').replace('_1', 'b').replace('_2', 'c').replace(
                    '_3', 'd').replace('_4', 'e').replace('_5', 'f')
                print(caffe_blob_name)
                caffe_net.params[caffe_blob_name][0].data[...] = caffe2kptmodel[caffe2kptmodel_key]

        elif 'fc' in caffe2kptmodel_key or 'cls_score' in caffe2kptmodel_key or 'bbox_pred' in caffe2kptmodel_key:
            biasOrWeight(caffe2kptmodel_key, caffe_net, caffe2kptmodel)

        #CAUTION!! blow layers shares same weights!!! (XXX_2)
        elif 'conv_rpn_fpn' in caffe2kptmodel_key:
            biasOrWeightWithSpecificLayer(caffe2kptmodel_key, caffe_net, 'conv_rpn_fpn2', caffe2kptmodel)

        elif 'rpn_cls_logits_fpn' in caffe2kptmodel_key:
            biasOrWeightWithSpecificLayer(caffe2kptmodel_key, caffe_net, 'rpn_cls_logits_fpn2', caffe2kptmodel)

        elif 'rpn_bbox_pred_fpn' in caffe2kptmodel_key:
            biasOrWeightWithSpecificLayer(caffe2kptmodel_key, caffe_net, 'rpn_bbox_pred_fpn2', caffe2kptmodel)

caffe_net.save('FRCNN-R50_RoiAlign_7x7.caffemodel')

if __name__ == '__main__':
    main()