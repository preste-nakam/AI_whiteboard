import cv2
import numpy as np
from hand_detector.yolo.darknet import model as yolo_model
from tensorflow.keras.models import load_model
from hand_detector.yolo.preprocess.yolo_flag import Flag as yoloFlag
from trt_utils import *


class YOLO:
    def __init__(self, weights, trt_engine, threshold, trt = False):
        self.f = yoloFlag()
        self.threshold = threshold
        self.trt = trt

        if self.trt:
            self.engine = load_engine(trt_engine)
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
            self.context = self.engine.create_execution_context()
        else:
            # self.model = load_model(model)
            self.model = yolo_model()
            self.model.load_weights(weights) 

    def detect(self, image):
        height, width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.f.target_size, self.f.target_size)) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # TensorRT engine  
        if self.trt:
            np.copyto(self.inputs[0].host, image.ravel())
            yolo_out = np.array([do_inference(self.context, 
            									bindings=self.bindings, 
            									inputs=self.inputs,       									
            									outputs=self.outputs, 
            									stream=self.stream)
            					]).reshape((1, 7, 7, 5))
        else:
            yolo_out = self.model.predict(image)

        yolo_out = yolo_out[0]
        grid_pred = yolo_out[:, :, 0]
        i, j = np.squeeze(np.where(grid_pred == np.amax(grid_pred)))
        
        try:
            if i.shape[0] > 1 :
                i = i[0]
                j = j[0]
        except:
            pass
        
        if grid_pred[i, j] >= self.threshold:
            bbox = yolo_out[i, j, 1:]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            # size conversion
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            return (x1, y1), (x2, y2)
        else:
            return None, None
