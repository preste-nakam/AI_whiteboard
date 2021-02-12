import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fingertips_detector.net.network import model
from trt_utils import *

class Fingertips:
    def __init__(self, weights, trt_engine, trt = False):
        
        self.trt = trt
        if self.trt:
            self.engine = load_engine(trt_engine)
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
            self.context = self.engine.create_execution_context()
        else:
            # self.model = load_model(model)
            self.model = model()
            self.model.load_weights(weights)
    @staticmethod
    def class_finder(prob):
        cls = ''
        classes = [0, 1, 2, 3, 4, 5, 6, 7]

        if np.array_equal(prob, np.array([0, 1, 0, 0, 0])):
            cls = classes[0]
        elif np.array_equal(prob, np.array([0, 1, 1, 0, 0])):
            cls = classes[1]
        elif np.array_equal(prob, np.array([0, 1, 1, 1, 0])):
            cls = classes[2]
        elif np.array_equal(prob, np.array([0, 1, 1, 1, 1])):
            cls = classes[3]
        elif np.array_equal(prob, np.array([1, 1, 1, 1, 1])):
            cls = classes[4]
        elif np.array_equal(prob, np.array([1, 0, 0, 0, 1])):
            cls = classes[5]
        elif np.array_equal(prob, np.array([1, 1, 0, 0, 1])):
            cls = classes[6]
        elif np.array_equal(prob, np.array([1, 1, 0, 0, 0])):
            cls = classes[7]
        return cls

    def classify(self, image):
        image = np.asarray(image)
        image = cv2.resize(image, (128, 128))
        image = image.astype('float32')
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        # TensorRT engine
        if self.trt:
            np.copyto(self.inputs[0].host, image.ravel())
            position, probability = do_inference(self.context, 
            									bindings=self.bindings, 
            									inputs=self.inputs, 												
            									outputs=self.outputs, 
            									stream=self.stream)
            					
            position = position.reshape((1,10,10))
            probability = probability.reshape((1,5))
        else:
            probability, position = self.model.predict(image)
            
        probability = probability[0]
        position = position[0]
        return probability, position
