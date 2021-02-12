import cv2
import numpy as np
import time
import copy
import argparse

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from fingertips_detector.unified_detector import Fingertips
from hand_detector.detector import YOLO
from config import config
from gst_cam import gstreamer_pipeline


class AIWhiteboard():
    """AI Whiteboard"""
    def __init__(self, args):
        """
        Initialization of AI Whiteboard class

        args.trt                     :boolean : if True - use TensorRT engines for inference
        args.raspberry_pi_camera     :boolean : if True - capture images from Raspberry Pi Camera
        """

        super(AIWhiteboard, self).__init__()
        self.confidence_ft_threshold = config['confidence_ft_threshold']
        self.confidence_hd_threshold = config['confidence_hd_threshold']
        self.colors = [(15, 15, 240),
                      (15, 240, 155),
                      (240, 155, 15),
                      (240, 15, 155),
                      (240, 15, 240)]

        # init models
        self.hand_detector = YOLO(weights='weights/trained_yolo.h5', 
                                  trt_engine = 'weights/engines/model_trained_yolo.fp16.engine', 
                                  threshold=self.confidence_hd_threshold, 
                                  trt = args.trt)

        self.fingertips_detector = Fingertips(weights='weights/classes8.h5', 
                                              trt_engine = 'weights/engines/model_classes8.fp16.engine', 
                                              trt = args.trt)
        if args.raspberry_pi_camera:
            self.cam = cv2.VideoCapture(gstreamer_pipeline(capture_width=config['cam_w'],
                                                           capture_height=config['cam_h'],
                                                           display_width=config['cam_w'],
                                                           display_height=config['cam_h'],
                                                           framerate=config['framerate']), 
                                        cv2.CAP_GSTREAMER)  
        else:
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, config['cam_w']) 
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config['cam_h'])


        origin_w  = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        origin_h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # cropped coordinates (to get a square image)
        self.cropped_x_st = int(origin_w/2) - int(origin_h/2)
        self.cropped_x_end = int(origin_w/2) + int(origin_h/2)

        # whiteboard_tl - top left corner of whiteboard on cropped image
        # whiteboard_br - bottom right corner of whiteboard on cropped image
        self.whiteboard_tl = (int((self.cropped_x_end-self.cropped_x_st-config['whiteboard_w'])/2), int((origin_h-config['whiteboard_h'])/2))
        self.whiteboard_br = (int((self.cropped_x_end-self.cropped_x_st+config['whiteboard_w'])/2), int((origin_h+config['whiteboard_h'])/2))
                
        # Create a whiteboard
        self.whiteboard = np.zeros((config['zoom_koef']*config['whiteboard_h'],
                                    config['zoom_koef']*config['whiteboard_w'],
                                    3), np.uint8) + 255
        # Create a info whiteboard for demonstration
        self.info_whiteboard = copy.deepcopy(self.whiteboard)


    def draw(self, prob, pos):
        """
        Draw detected fingers on whiteboard

        prob :numpy array : array of confidance score of each finger according to Fingertips detector
        pos  :numpy array : array of relative fingers position on whiteboard according to Fingertips detector
        """

        # whiteboard shape
        width = config['whiteboard_w'] * config['zoom_koef']
        height = config['whiteboard_h'] * config['zoom_koef']

        # number of detected fingers
        n_fingers = int(np.sum(prob))

        # one finger detected : INDEX  | action: paint
        if n_fingers == 1 and prob[1] == 1.0:
            center = (int(pos[2]*width), int(pos[3]*height) )
            cv2.circle(self.whiteboard, center, radius=5, color=(0,0,0), thickness=-1)

            self.info_whiteboard = copy.deepcopy(self.whiteboard)
            cv2.circle(self.info_whiteboard, center, radius=5, color=(0,20,200), thickness=2)
        
        # two fingers detected: THUMB + INDEX | action: show pointer
        elif n_fingers == 2 and prob[1] == 1.0 and prob[0] == 1.0:
            center = (int(pos[2]*width), int(pos[3]*height) )
            
            self.info_whiteboard = copy.deepcopy(self.whiteboard)
            cv2.circle(self.info_whiteboard, center, radius=5, color=(255,0,0), thickness=2)
        
        # five fingers detected | action:  erase 
        elif n_fingers == 5 :
            center = (int(pos[2]*width), int(pos[3]*height) )
            cv2.circle(self.whiteboard, center, radius=10, color=(255,255,255), thickness=-1)

            self.info_whiteboard = copy.deepcopy(self.whiteboard)
            cv2.circle(self.info_whiteboard, center, radius=12, color=(0,255,0), thickness=2)

        # two fingers detected: THUMB + PINKY | action: clean whiteboard
        elif n_fingers == 2 and prob[0] == 1.0 and prob[4] == 1.0:
            self.whiteboard = np.zeros((height,width,3), np.uint8) + 255
            self.info_whiteboard = copy.deepcopy(self.whiteboard)
        
        # three fingers detected: THUMB + MIDDLE + RING | action: save whiteboard
        elif n_fingers == 3 and prob[1] == 1.0 and prob[2] == 1.0 and prob[3] == 1.0:
            cv2.imwrite('saved/whiteboard.jpg', self.whiteboard)
            print('-- whiteboard.jpg saved! ')
            self.info_whiteboard = copy.deepcopy(self.whiteboard)

        # three fingers detected: THUMB + INDEX + PINKY | action: exit
        # elif n_fingers == 3 and prob[0] == 1.0 and prob[1] == 1.0 and prob[4] == 1.0:
        #   info_whiteboard = copy.deepcopy(whiteboard)
        #   k = 1
        #   print('=== EXIT ===')
        else:
            self.info_whiteboard = copy.deepcopy(self.whiteboard)
        

    def run(self):
        """
        Run AI Whiteboard 
        """
        try:
            while True:
                ret, image = self.cam.read()
                image = image[:,self.cropped_x_st:self.cropped_x_end,:]

                if ret is False:
                    break

                start = time.time()

                # hand detection
                # tl - top left corner of hand bbox on cropped image
                # br - bottom right corner of hand bbox on cropped image
                tl, br = self.hand_detector.detect(image=image)
                if tl and br is not None and br[0] - tl[0] >= 5 and  br[1] - tl[1] >= 5:
                    cropped_hand = image[tl[1]:br[1], tl[0]: br[0]]
                    height_hand, width_hand, _ = cropped_hand.shape

                    # gesture classification and fingertips regression
                    prob, pos = self.fingertips_detector.classify(image=cropped_hand)
                    pos = np.mean(pos, 0)

                    # post-processing: absolute fingers position on an image
                    prob = np.asarray([(p >= self.confidence_ft_threshold) * 1.0 for p in prob])
                    for i in range(0, len(pos), 2):
                        pos[i] = pos[i] * width_hand + tl[0]
                        pos[i + 1] = pos[i + 1] * height_hand + tl[1]

                    # post-processing: relative fingers position on a whiteboard
                    relative_pos = []
                    for i in range(0, len(pos), 2):
                        tmp_x = max(-5, pos[i] - self.whiteboard_tl[0])/config['whiteboard_w']
                        tmp_y = max(-5, pos[i+1] - self.whiteboard_tl[1])/config['whiteboard_h']
                        relative_pos.append(tmp_x)
                        relative_pos.append(tmp_y)
                    relative_pos = np.array(relative_pos)
                    # draw on whiteboard 
                    self.draw(prob, relative_pos)
                    
                    # drawing fingertips
                    index = 0
                    for c, p in enumerate(prob):
                        if p >= self.confidence_ft_threshold:
                            image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=5,
                                               color=self.colors[c], thickness=-2)
                        index += 2

                k = cv2.waitKey(1)
                if k==27:       # Esc key to stop
                    break

                end = time.time()

                str_fps = '{:.1f} fps'.format(1/(end-start))
                # print(str_fps)
                cv2.putText(image, str_fps,(15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2,cv2.LINE_AA)
                image = cv2.rectangle(image, (self.whiteboard_tl[0], self.whiteboard_tl[1]), (self.whiteboard_br[0], self.whiteboard_br[1]), (255, 255, 255), 2)
                
                # display image
                cv2.imshow('Fingertips', cv2.resize(image, (config['zoom_koef']*config['whiteboard_h'],config['zoom_koef']*config['whiteboard_w'])))
                # display whiteboard
                cv2.imshow('AI_whiteboard', self.info_whiteboard)


            self.cam.release()
            cv2.destroyAllWindows()

        except Exception as e:
            self.cam.release()
            cv2.destroyAllWindows()
            print("Error: {}".format(e))
            exit(1)


def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Whiteboard arguments')
    
    parser.add_argument('--rpc', dest='raspberry_pi_camera', action='store_true', help='Run AI whiteboard with Raspberry Pi Camera')
    parser.set_defaults(raspberry_pi_camera=False)
    parser.add_argument('--trt', dest='trt', action='store_true', help='Use TensoRT engine')
    parser.set_defaults(trt=False)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ai_w = AIWhiteboard(args)
    ai_w.run()
