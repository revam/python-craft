"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import argparse
import craft_utils
import time
import torch

from craft import CRAFT
from refinenet import RefineNet
from refined_craft import RefinedCRAFT

def export_net(net, canvas_size, export_name):
    x = torch.randn(1, 3, canvas_size, canvas_size)

    # export
    with torch.no_grad():
        torch.onnx.export(
            net,
            x,
            export_name,
            verbose=True,
            input_names=['images'],
            output_names=['text_matrix', 'link_matrix'],
            dynamic_axes={
                'images': {0: 'batch size', 2: 'image width', 3: 'image height'},
                'text_matrix': {0: 'batch size', 1: 'matrix width', 2: 'matrix height'},
                'link_matrix': {0: 'batch size', 1: 'matrix width', 2: 'matrix height'},
            }
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    # model arguments
    parser.add_argument('--trained_model', default='./weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--refiner_model', default='./weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    # pre-processing arguments
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')

    # test arguements
    parser.add_argument('--export_name', default='./refined_craft.onnx', type=str, help='name of the exported onnx format model')

    args = parser.parse_args()

    t0 = time.time()

    # create base net and load weights
    base_net = CRAFT()
    print('Loading weights of base net from checkpoint (' + args.trained_model + ')')
    base_net.load_state_dict(craft_utils.copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    base_net.eval()

    # create refine net and load weights
    refine_net = RefineNet()
    print('Loading weights of refine net from checkpoint (' + args.refiner_model + ')')
    refine_net.load_state_dict(craft_utils.copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
    refine_net.eval()

    # create combined net
    net = RefinedCRAFT(base_net, refine_net)
    net.eval()

    t0 = time.time() - t0
    t1 = time.time()

    export_net(net, args.canvas_size, args.export_name)

    t1 = time.time() - t1
    print("\nload/export time: {:.3f}/{:.3f}".format(t0, t1))
