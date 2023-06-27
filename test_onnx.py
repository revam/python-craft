"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import argparse
import craft_utils
import cv2
import file_utils
import imgproc
import os
import time

import numpy as np
from onnxruntime import InferenceSession

def test_session(session: InferenceSession, image, text_threshold, link_threshold, low_text, canvas_size, mag_ratio):
    t0 = time.time()

    # resize
    img_resized, target_ratio, target_w, target_h, _ = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = np.transpose(x, (2, 0, 1)) # [h, w, c] to [c, h, w]
    x = np.expand_dims(x, axis=0) # [c, h, w] to [b, c, h, w]
    inputs = { "images": x }

    t0 = time.time() - t0
    t1 = time.time()

    # forward pass
    outputs = session.run(None, inputs)
    score_text = outputs[0][0,:,:]
    score_link = outputs[1][0,:,:]

    t1 = time.time() - t1
    t2 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetectionBoxesAndPolygons(score_text, score_link, text_threshold, link_threshold, low_text)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t2 = time.time() - t2

    # render results (optional)
    ret_score_text = imgproc.cvt2HeatmapImg(score_text.copy()[0:target_w, 0:target_h])
    ret_score_link = imgproc.cvt2HeatmapImg(score_link.copy()[0:target_w, 0:target_h])

    print("\npreprocess/infer/postprocess time : {:.3f}/{:.3f}/{:.3f}".format(t0, t1, t2))

    return boxes, polys, ret_score_text, ret_score_link

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    # model arguments
    parser.add_argument('--model', default='./refined_craft.onnx', type=str, help='exported ONNX model')

    # pre-processing arguments
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')

    # post-processing arguments
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')

    # test arguements
    parser.add_argument('--image_folder', default='./data/', type=str, help='path to the images input folder')
    parser.add_argument('--result_folder', default='./result/', type=str, help='path to output folder')

    args = parser.parse_args()

    t0 = time.time()

    # load the ONNX model
    print('Loading inference session (' + args.model + ')')
    session = InferenceSession(args.model, providers=['CPUExecutionProvider'])

    t0 = time.time() - t0
    t1 = time.time()

    # For test images in a folder
    image_list = file_utils.get_files(args.image_folder)

    if not os.path.isdir(args.result_folder):
        os.mkdir(args.result_folder)

    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text, score_link = test_session(session, image, args.text_threshold, args.link_threshold, args.low_text, args.canvas_size, args.mag_ratio)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        text_file = args.result_folder + "/onnx_" + filename + '_text.jpg'
        link_file = args.result_folder + "/onnx_" + filename + '_link.jpg'
        cv2.imwrite(text_file, score_text)
        cv2.imwrite(link_file, score_link)

        file_utils.saveResult("/onnx" + filename, image[:,:,::-1], polys, dirname=args.result_folder)

    t1 = time.time() - t1
    print("\nload/test time: {:.3f}/{:.3f}".format(t0, t1))
