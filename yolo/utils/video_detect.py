import cv2
import time 
import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from yolo.modeling.yolo_v3 import Yolov3, DarkNet53
import yolo.modeling.building_blocks as nn_blocks
import datetime
import tensorflow.keras.backend as K
import colorsys

def draw_box(image, boxes, classes, conf, colors, label_names):
    for i in range(boxes.shape[0]):
        if boxes[i][3] == 0:
            break
        box = boxes[i]
        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes[i]], 1)
        cv2.putText(image, "%s, %0.3f"%(label_names[classes[i]], conf[i]), (box[0], box[2]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[classes[i]], 1)
    return

def int_scale_boxes(boxes, classes, width, height):
    boxes = K.stack([tf.cast(boxes[..., 1] * width, dtype = tf.int32),tf.cast(boxes[..., 3] * width, dtype = tf.int32), tf.cast(boxes[..., 0] * height, dtype = tf.int32), tf.cast(boxes[..., 2] * height, dtype = tf.int32)], axis = -1)
    classes = tf.cast(classes, dtype = tf.int32)
    return boxes, classes

def gen_colors(max_classes):
    hue = np.linspace(start = 0, stop = 1, num = max_classes)
    np.random.shuffle(hue)
    colors = []
    for val in hue:
        colors.append(colorsys.hsv_to_rgb(val, 0.75, 1.0))
    return colors

def get_coco_names(path = "/home/vishnu/Desktop/CAM2/TensorFlowModelGardeners/yolo/dataloaders/dataset_specs/coco.names"):
    f = open(path, "r")
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i][:-1]
    return data

'''Video Buffer using cv2'''
def video_processor(vidpath):
    cap = cv2.VideoCapture(vidpath)
    assert cap.isOpened()
    width = 0
    height = 0
    frame_count = 0
    img_array = []


    width = int(cap.get(3))
    height = int(cap.get(4))
    print('width, height, fps:', width, height, int(cap.get(5)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(5, 30)

    i = 0
    t = 0
    start = time.time()
    tick = 0
    e,f,a,b,c,d = 0,0,0,0,0,0
    with tf.device("/GPU:0"): 
        model = build_model()
        model.make_predict_function()
    
    colors = gen_colors(80)
    label_names = get_coco_names()
    print(label_names)

    # output_writer = cv2.VideoWriter('yolo_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_count, (480, 640))  # change output file name if needed
    # may be batch by 2?
    pred = None
    while i <= frame_count:
        success, image = cap.read()

        with tf.device("/CPU:0"): 
            e = datetime.datetime.now()
            image = tf.cast(image, dtype = tf.float32)
            image = image/255
            f = datetime.datetime.now()

        if t % 1 == 0:
            a = datetime.datetime.now()
            with tf.device("/GPU:0"):
                pimage = tf.expand_dims(image, axis = 0)
                pimage = tf.image.resize(pimage, (416, 416))
                pred = model.predict(pimage)
            b = datetime.datetime.now()

        image = image.numpy()
        if pred != None:
            c = datetime.datetime.now()
            boxes, classes = int_scale_boxes(pred[0], pred[1], width, height)
            draw_box(image, boxes[0].numpy(), classes[0].numpy(), pred[2][0], colors, label_names)
            d = datetime.datetime.now()

        cv2.imshow('frame', image)
        i += 1   
        t += 1  

        if time.time() - start - tick >= 1:
            tick += 1
            print(i, end = "\n")
            print(f"pred time: {(f - e) * 1000} ms")
            print(f"pred time: {(b - a) * 1000} ms")
            print(f"draw time: {(d - c) * 1000} ms")
            i = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return

def webcam():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    cap.set(3, 416)
    cap.set(4, 416)
    i = 0
    start = time.time()
    tick = 0
    with tf.device("/GPU:0"): 
        model = build_model()
        model.make_predict_function()
    while(True):
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Capture frame-by-frame
        ret, image = cap.read()

        # with tf.device("/GPU:0"): 
        #     image = tf.cast(image, dtype = tf.float32)
        #     image = tf.image.resize(image, (416, 416))
        #     image = image/255
        #     image = tf.expand_dims(image, axis = 0)
        #     pred = model.predict(image)
        #     image = tf.image.draw_bounding_boxes(image, pred[0][0], [[0.0, 0.0, 1.0]])
        #     image = image[0]    

        cv2.imshow('frame', image)#.numpy())
        i += 1     

        # print(time.time() - start)   

        if time.time() - start - tick >= 1:
            tick += 1
            print(i)
            i = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def build_model():
    #build backbone without loops
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    # using mixed type policy give better performance than strictly float32
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    w = 416
    h = 416

    model = Yolov3(classes = 80, boxes = 9, type = "regular", input_shape=(1, w, h, 3))
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file="yolov3-regular.weights")
    model.summary()

    inputs = ks.layers.Input(shape=[w, h, 3])
    outputs = model(inputs) 
    b1, c1 = nn_blocks.YoloFilterCell(anchors = [(116,90),  (156,198),  (373,326)], thresh = 0.5, dtype=policy.compute_dtype)(outputs[1024])
    b2, c2 = nn_blocks.YoloFilterCell(anchors = [(30,61),  (62,45),  (59,119)], thresh = 0.5, dtype=policy.compute_dtype)(outputs[512])
    b3, c3 = nn_blocks.YoloFilterCell(anchors = [(10,13),  (16,30),  (33,23)], thresh = 0.5, dtype=policy.compute_dtype)(outputs[256])
    b = K.concatenate([b1, b2, b3], axis = 1)
    c = K.concatenate([c1, c2, c3], axis = 1)

    # b1, c1 = nn_blocks.YoloFilterCell(anchors = [(81,82),  (135,169),  (344,319)], thresh = 0.5, dtype=policy.compute_dtype)(outputs[1024])
    # b2, c2 = nn_blocks.YoloFilterCell(anchors = [(10,14),  (23,27),  (37,58)], thresh = 0.5, dtype=policy.compute_dtype)(outputs[256])
    # b = K.concatenate([b1, b2], axis = 1)
    # c = K.concatenate([c1, c2], axis = 1)
    nms = tf.image.combined_non_max_suppression(tf.expand_dims(b, axis=2), c, 100, 100, 0.5, 0.5)
    # outputs = nn_blocks.YoloLayer(masks = {1024:[6, 7, 8], 512:[3,4,5] ,256:[0,1,2]}, 
    #                              anchors =[(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)], 
    #                              thresh = 0.5)(outputs) # -> 1 frame cost
    # outputs = nn_blocks.YoloLayer(masks = {1024:[3,4,5],256:[0,1,2]}, 
    #                     anchors =[(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)], 
    #                     thresh = 0.5)(outputs)
    # run = ks.Model(inputs = [inputs], outputs = [outputs])
    run = ks.Model(inputs = [inputs], outputs = [nms.nmsed_boxes,  nms.nmsed_classes, nms.nmsed_scores])
    run.build(input_shape = (1, w, h, 3))
    run.summary()
    return run

def main():
    # vid_name = "yolo_vid.mp4"  # change input name if needed
    # video_processor(vid_name)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    video_processor("nyc.mp4")
    return 0


if __name__ == "__main__":
    main()
