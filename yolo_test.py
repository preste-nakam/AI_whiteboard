import pandas as pd
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import cv2
from trt_utils import *
from tensorflow.keras.models import load_model
from hand_detector.yolo.darknet import model as yolo_model
from hand_detector.yolo.generator import load_test_images
from hand_detector.yolo.preprocess.yolo_flag import Flag
from metrics import iou, get_stat

f = Flag()
# TEST DATASET LABELS
df_test = pd.read_csv('custom_dataset/test_labels.csv') 


def get_test_image(image_name, directory = 'custom_dataset/'):
    image = cv2.imread(directory + 'test/' + image_name, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (f.target_size, f.target_size))
    processed_image = np.expand_dims(image, axis=0) / 255.0
    return processed_image


def get_test_bbox(image_name):

    label = df_test[df_test.filename == image_name[:-4]].iloc[0][1:].tolist()
    bbox = [float(element) * f.target_size for element in label]
    bbox = tuple(bbox)
    if bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0:
        return None
    return bbox


def convert_anchor_to_bbox(yolo_out, threshold = 0.8, width=224, height=224):

    grid_pred = yolo_out[:, :, 0]
    i, j = np.squeeze(np.where(grid_pred == np.amax(grid_pred)))

    try:
        if i.shape[0] > 1 :
            i = i[0]
            j = j[0]
    except:
        pass

    if grid_pred[i, j] >= threshold:
        bbox = yolo_out[i, j, 1:]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        # size conversion
        x1 = float(x1 * width)
        y1 = float(y1 * height)
        x2 = float(x2 * width)
        y2 = float(y2 * height)
        return (x1, y1, x2, y2)
    else:
        return None


# def show_result(preprocess, pr_bbox, gt_bbox, tmp_iou):
#     image = preprocess.astype(np.float32)
#     if pr_bbox is not None:
#         x1, y1, x2, y2 = int(pr_bbox[0]), int(pr_bbox[1]), int(pr_bbox[2]), int(pr_bbox[3])
#         image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,0), 2)
#     if gt_bbox is not None:
#         x1, y1, x2, y2 = int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[2]), int(gt_bbox[3])
#         image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
# 
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     cv2.putText(image, '{:.2f}'.format(tmp_iou), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
#     cv2.imshow('test_image', image)


def run_test(weights = 'weights/yolo.h5', trt_engine = 'weights/engines/model_trained_yolo.fp16.engine', iou_threshold = 0.5, confidence_threshold = 0.8, trt = False, show = True):

  
    if trt: 
        engine = load_engine(trt_engine)
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        context = engine.create_execution_context() 
    else:
        # create the model
        model = yolo_model()
        model.load_weights(weights)
    
    # model.summary()

    # test
    list_test_images = load_test_images()
    test_set_size = len(list_test_images)
    print('Test_set_size : ', test_set_size)

    iou_list = []
    pr_list = []
    gt_list = []

    for i in range(test_set_size):
        print(i)
        image_name = list_test_images[i]
        preprocess = get_test_image(image_name)
        np.copyto(inputs[0].host, preprocess.ravel())
        if trt:
            yolo_out = np.array([do_inference(context, 
            									bindings=bindings, 
            									inputs=inputs,       									
            									outputs=outputs, 
            									stream=stream)
            					]).reshape((1, 7, 7, 5))
            yolo_output = yolo_out[0]
        else:    					
            yolo_output = model.predict(preprocess)[0]
       
        pr_bbox = convert_anchor_to_bbox(yolo_output, threshold = confidence_threshold, width=f.target_size, height=f.target_size)
        gt_bbox = get_test_bbox(image_name)


        if gt_bbox is None and pr_bbox is None:
            pr_list.append(0)
            gt_list.append(0)
            tmp_iou = -1

        elif gt_bbox is None and pr_bbox is not None:
            pr_list.append(1)
            gt_list.append(0)
            iou_list.append(0)
            tmp_iou = 0

        elif gt_bbox is not None and pr_bbox is None:
            pr_list.append(0)
            gt_list.append(1)
            iou_list.append(0)
            tmp_iou = 0
        elif gt_bbox is not None and pr_bbox is not None:
            gt_list.append(1)
            tmp_iou = iou(gt_bbox, pr_bbox)
            
            if tmp_iou > iou_threshold:
                pr_list.append(1)
            else:
                pr_list.append(0)

            iou_list.append(tmp_iou)
        
        #if show:
        #    show_result(preprocess[0], pr_bbox, gt_bbox, tmp_iou)
        #    if cv2.waitKey(60) & 0xff == 27:
        #        cv2.destroyAllWindows()
        #        break
    avg_iou = sum(iou_list)/len(iou_list)
    acc, recall, precision, _ = get_stat(gt_list, pr_list)

    print('Avg iou   : {:.2f}'.format(avg_iou*100))
    print('Accuracy  : {:.2f} %'.format(acc*100))
    print('Recall    : {:.2f} %'.format(recall*100))
    print('Precision : {:.2f} %'.format(precision*100))

if __name__ == '__main__':
    print('\n\n --------- yolo -----------')
    run_test(weights = 'weights/yolo.h5',  trt_engine = 'weights/engines/model_trained_yolo.fp32.engine', iou_threshold = 0.5, confidence_threshold = 0.8, trt = True)
