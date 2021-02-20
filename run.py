#!/usr/bin/env python

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os, sys, glob, argparse, numpy as np, PIL.Image as pil, matplotlib as mpl, matplotlib.cm as cm, torch, rospy, cv2
from torchvision import transforms, datasets
from sensor_msgs.msg import Image

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from cv_bridge import CvBridge, CvBridgeError


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument('--image', type=str, default='/camera/image_raw', help='Image topic input')
    parser.add_argument('--out', type=str, default='/camera/image_depth', help='Disparity image topic output')
    parser.add_argument('--model', type=str, default='/home/rhidra/catkin_ws/src/ros-monodepth2/models/mono_640x192')
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def run_node(args):
    """Function to predict for a single image or folder of images
    """
    device = torch.device("cuda") if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")

    download_model_if_doesnt_exist(args.model)
    model_path = args.model
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    node = rospy.init_node('monodepth2', anonymous=True)
    pub_img = rospy.Publisher(args.out, Image, queue_size=1)
    pub_depth = rospy.Publisher('/image_rect', Image, queue_size=1)
    bridge = CvBridge()

    def process_image(img_msg):
        with torch.no_grad():
            try:
                cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {}".format(e))
                return

            # Load image and preprocess
            input_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            input_image = pil.fromarray(input_image)
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            disp_resized = disp_resized.squeeze().cpu().numpy()
            _, depth = disp_to_depth(disp_resized, 10, 100)

            # Saving colormapped depth image
            vmax = np.percentile(disp_resized, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            # Publish in ROS topic
            msg = Image()
            msg.header.stamp = rospy.Time.now()
            msg.height = im.height
            msg.width = im.width
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = 3 * im.width
            msg.data = np.array(im).tobytes()
            pub_img.publish(msg)

            depth = pil.fromarray(np.array(depth))
            depth = depth.convert('L')
            msg = Image()
            msg.header.stamp = rospy.Time.now()
            msg.height = depth.height
            msg.width = depth.width
            msg.encoding = "mono8"
            msg.is_bigendian = False
            msg.step = 1 * depth.width
            msg.data = np.array(depth).tobytes()
            pub_depth.publish(msg)

    rospy.Subscriber(args.image, Image, process_image, queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    args = parse_args()
    run_node(args)
