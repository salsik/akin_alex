from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import spatial, stats

from PIL import Image as im

from PIL import ImageDraw, ImageFont

from datetime import datetime

import argparse
import json
import os




#from api_models import (

from src.api_models import (
    DimensionSchema,
    ObjectsSchema,
    PositionSchema,
    UIDesignPattern,
    WireframeSchema,
)
from src.sagan_models import create_generator

#from sagan_models import create_generator


GEN = create_generator(image_size=128, z_dim=128, filters=16, kernel_size=4)

#GEN.load_weights("/checkpoints/g/cp-007000.ckpt")

#GEN.load_weights("./checkpoint_main/g/cp-007000.ckpt")
#GEN.load_weights("/data1/data_alex/rico/akin experiments/exp3 tanh lr 0.0005 0.0008 0.2 0.9 0.5/g tanh/cp-007950.ckpt")
GEN.load_weights("/home/atsumilab/alex/rico/akin-generator/checkpoint/20230124-221114 lr 0.0001 0.0004 0.0 0.9 10/g/cp-007300.ckpt")

color_map_file = Path("resources/ui_labels_color_map.csv")
color_map = pd.read_csv(color_map_file, index_col=0, header=None)
color_np_list = color_map.to_numpy()
labels = color_map.index.values

kdt = spatial.KDTree(color_np_list)


def resize_screen(s, interpolation):
    image = ((s[:, :, ::-1] + 1) * 127).astype(np.uint8)
    return cv2.resize(
        image,
        (360, 576),
        interpolation=interpolation,
    )


def sub_threshold(img, erode_flag=False, unsharp_flag=False):
    if unsharp_flag:
        img = unsharp(img)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if erode_flag:
        thresh = erode(thresh)
    return thresh


def threshold(img):
    m1 = sub_threshold(img[:, :, 0], True, True)
    m2 = sub_threshold(img[:, :, 1], True, True)
    m3 = sub_threshold(img[:, :, 2], True, True)

    res = cv2.add(m1, cv2.add(m2, m3))
    return res


def erode(thresh):
    kernel = np.ones((3, 4), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=3)
    return thresh


def unsharp(imgray):
    imgray = imgray.copy()
    gaussian = cv2.GaussianBlur(imgray, (7, 7), 10.0)
    unsharp_image = cv2.addWeighted(imgray, 2.5, gaussian, -1.5, 0, imgray)
    return unsharp_image


def get_nearest_dominant_color(img):

    pixels = img.reshape(-1, 3)
    if len(pixels) < 50:
        return None, None
    _, ind = kdt.query(pixels)
    m = stats.mode(ind)
    closest_color = color_np_list[m[0][0]]
    label = labels[m[0][0]]
    return (int(closest_color[0]), int(closest_color[1]), int(closest_color[2])), label


def get_wireframe(i, image, category):
    original = image.copy()
    thresh = threshold(image)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    objects = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 25 or h < 25:
            continue
        ROI = original[y : y + h, x : x + w]
        dominant_color, label = get_nearest_dominant_color(ROI)

        if label == "name" and category == UIDesignPattern.product_listing:
            label = "filter"
        if label == "filter" and category == UIDesignPattern.splash:
            label = "sign_up"
        if label == "rating" and category == UIDesignPattern.splash:
            label = "image"
        if label == "sort" and category == UIDesignPattern.splash:
            label = "button"

        if dominant_color is None:
            continue

        position = PositionSchema(x=x, y=y)
        dimension = DimensionSchema(width=w, height=h)
        element = ObjectsSchema(name=label, position=position, dimension=dimension)
        # WireframeSchema
        objects.append(element)

    height, width, _ = original.shape

    wireframe: WireframeSchema = WireframeSchema(
        id=str(i), width=width, height=height, objects=objects
    )

    return wireframe



def get_bounding_boxes(dir, image_name, dst_path, dir_name):
    elements = []
    image = cv2.imread(os.path.join(dir, image_name))
    original = image.copy()
    thresh = threshold(image)
    new_semantic = np.ones_like(original) * 255
    # Find contours, obtain bounding box, extract and save ROI
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        ROI = original[y:y + h, x:x + w]
        dominant_color, label = get_nearest_dominant_color(ROI)
        if dominant_color is None:
            continue
        cv2.rectangle(new_semantic, (x, y), (x + w, y + h), dominant_color, 3)
        elements.append({"points": [[x, y], [x + w, y + h]], "label": label})
    print(os.path.join(dst_path, image_name[:-4] + "0.png"))
    cv2.imwrite(os.path.join(dst_path, image_name[:-4] + "_0.png"), image)
    cv2.imwrite(os.path.join(dst_path, image_name[:-4] + "_1.png"), new_semantic)

    # creating json file

    json_file =os.path.join(".", image_name[:-4] + ".json")
    create_json_file(json_file, elements, dir_name)


    # convert to wireframe 

    print(json_file)

    dst_file_path = os.path.join(".", image_name[:-4] +"wire 1.jpg")
    # dst_file_path = os.path.join(dst_folder, file[:-5]+".jpg")
    elements = get_elements(json_file, False)
    print(dst_file_path)
    
    create_img(elements, dst_file_path, dir)
    #try:
        #create_img(elements, dst_file_path, dir)
        #print(file, count)
    #except Exception as e:
    #    print(e, dst_file_path)
    



def load_all_ui_images():
    android_label_map = {}
    android_element_mapping_file = "./resources/ui_labels_android_map.csv"
    android_elements_base_path = "./resources/android_elements"

    with open(android_element_mapping_file, "r") as f:
        data = f.readlines()
        for line in data:
            s = line.split(";")
            label = s[0]
            img_name = s[1]
            if len(img_name) > 0:
                img_path = os.path.join(android_elements_base_path, img_name + ".jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(android_elements_base_path, img_name + ".png")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            else:
                img = None
            text = s[2]
            if text is None or len(text) == 0:
                text = None
            else:
                text = text.strip().split(",")
            resize = int(s[3])
            android_label_map[label] = {"img": img, "text": text, "resize": resize, "label": label}
    return android_label_map



def element_resize(img, w, h, flag, base_shade):
    """

    :param img:
    :param flag: 0 -> no
                1 -> normal
                2 -> maintain aspect ratio with center align
                3 -> maintain aspect ratio with left align
                4 -> maintain aspect ratio with right align
                5 -> normal but double text
    :return:
    """
    if flag == 0:
        return img, 0
    elif flag == 1 or flag == 5:
        return cv2.resize(img, (w, h)), 0
    elif flag == 2 or flag == 3 or flag == 4:
        label_image = np.ones((h, w, 3)) * base_shade
        ih = img.shape[0]
        iw = img.shape[1]
        rw = w / iw
        rh = h / ih
        fw = 0
        if rw == 1 and rh == 1:
            label_image = img
        elif rw < rh:
            fw = reshape_to_w(h, ih, img, iw, label_image, w, flag)
        else:
            fw = reshape_to_h(h, ih, img, iw, label_image, w, flag)
        return label_image, fw


def element_resize_old(img, w, h, flag, base_shade):
    """

    :param img:
    :param flag: 0 -> no
                1 -> normal
                2 -> maintain aspect ratio with center align
                3 -> maintain aspect ratio with left align
                4 -> maintain aspect ratio with right align
                5 -> normal but double text
    :return:
    """
    if flag == 0:
        return img, 0
    elif flag == 1 or flag == 5:
        return cv2.resize(img, (w, h)), 0
    elif flag == 2 or flag == 3 or flag == 4:
        label_image = np.ones((h, w, 3)) * base_shade
        ih = img.shape[0]
        iw = img.shape[1]
        dw = w - iw
        dh = h - ih
        fw = 0
        if dw == 0 and dh == 0:
            label_image = img
        elif dw > 0 and dh > 0:
            if dw < dh:
                fw = reshape_to_w(h, ih, img, iw, label_image, w, flag)
            else:
                fw = reshape_to_h(h, ih, img, iw, label_image, w, flag)
        elif dw <= 0 and dh >= 0:
            fw = reshape_to_w(h, ih, img, iw, label_image, w, flag)
        elif dw >= 0 and dh <= 0:
            fw = reshape_to_h(h, ih, img, iw, label_image, w, flag)
        elif dw < 0 and dh < 0:
            rw = w / iw
            rh = h / ih
            if rw < rh:
                fw = reshape_to_w(h, ih, img, iw, label_image, w, flag)
            else:
                fw = reshape_to_h(h, ih, img, iw, label_image, w, flag)
        return label_image, fw


def reshape_to_h(h, ih, img, iw, label_image, w, align):
    r = h / ih
    tw = int(iw * r)
    img = cv2.resize(img, (tw, h))
    if align == 2:  # center
        t = int((w - tw) / 2)
        label_image[0:h, t : t + tw] = img
    elif align == 3:  # left
        label_image[0:h, 0:tw] = img
    elif align == 4:  # right
        label_image[0:h, w - tw - 1 : w - 1] = img
    return tw


def reshape_to_w(h, ih, img, iw, label_image, w, align):
    r = w / iw
    th = int(ih * r)
    img = cv2.resize(img, (w, th))
    if align == 2:  # center
        t = int((h - th) / 2)
        label_image[t : t + th, 0:w] = img
    elif align == 3:  # left
        label_image[0:th, 0:w] = img
    elif align == 4:  # right
        label_image[h - th - 1 : h - 1, 0:w] = img
    return w



def find_font_scale_pil(fontScale, h, label_text, w, reduce_text):
    given_fontScale = fontScale
    
    fontpath = "./resources/fonts/DroidSans.ttf"

    font = ImageFont.truetype(fontpath, fontScale)
    textsize = font.getsize(label_text)
    tw = textsize[0]
    th = textsize[1]
    font_scale_reduction = 0
    while fontScale > 1 and (th > h * 0.90 or tw > w * 0.85):
        fontScale -= 1
        font_scale_reduction += 1
        if reduce_text and font_scale_reduction > 5:
            label_text, reduced = reduce_text_size(label_text)
            font_scale_reduction = 0
            if reduced:
                fontScale = given_fontScale
        font = ImageFont.truetype(fontpath, fontScale)
        textsize = font.getsize(label_text)
        th = textsize[1]
        tw = textsize[0]
    return font, textsize, label_text, fontScale


def reduce_text_size(text):
    if len(text) - 4 >= 5:
        new_length = len(text) - 4
        r = text[0:new_length]
        return r, True
    else:
        return text, False


def add_text_pil(label_image, label_text, align, label_resize, fw):
    level = 3
    default_fontScale = 20
    if label_text is not None and len(label_text) > 0:
        reduce_text = False
        if label_text == "lorem ipsum dolor":
            reduce_text = True
        if "/\\" in label_text:
            fontScale = default_fontScale - 4
            reduce_text = True
        else:
            fontScale = default_fontScale
        label_image = label_image.astype(np.uint8)
        h = label_image.shape[0]
        w = label_image.shape[1]
        if label_resize == 3 and fw > 0 and w > fw:
            accesible_w = w - fw
        else:
            accesible_w = w
        font, textsize, label_text, fontScale = find_font_scale_pil(fontScale, h, label_text, accesible_w, reduce_text)
        if label_resize == 3 and fontScale < 12:
            return label_image, True
        if align == 0:  # center
            textX = int(((label_image.shape[1] - fw) - textsize[0]) / 2) + fw
            textY = int((label_image.shape[0] - textsize[1]) / 2)
            img_pil = im.fromarray(label_image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((textX, textY - 1), label_text, font=font, fill=(51, 51, 51, 0))
            label_image = np.array(img_pil)
    return label_image, False



def create_img(elements, dst_file_path, cat, real=False):

    img_w = 360
    img_h = 576
    #json_location = args.json_file_location
    #android_element_mapping_file = args.android_element_mapping_file
    #android_elements_base_path = args.android_elements_base_path
    #dst_folder = args.destination_folder

    android_label_map = load_all_ui_images()


    base_image = np.ones((img_h, img_w, 3)) * 255
    elements.sort(key=lambda x: (x[1][1], x[1][0]))
    element_counted = {}
    for label, bb in elements:
        x1 = int(bb[0])
        y1 = int(bb[1])
        x2 = int(bb[2]) - 1
        y2 = int(bb[3]) - 1
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        if x1 >= 0 and y1 >= 0 and x2 < img_w and y2 < img_h and w > 0 and h > 0:
            if not real and (h < 20 or w < 20):
                continue
            elif h <= 0 or w <= 0 or y >= img_h or x >= img_w:
                continue
            if label == "name" and cat == "product_listing":
                label = "filter"
            if label == "filter" and cat == "splash":
                label = "sign_up"
            if label == "rating" and cat == "splash":
                label = "image"
            if label == "sort" and cat == "splash":
                label = "button"
            label_image = android_label_map[label]["img"]
            if label_image is None:
                continue
            base_label_image = label_image.copy()
            label_text = android_label_map[label]["text"]
            label_resize = android_label_map[label]["resize"]
            if label in ["navigation_dots", "sort, heart_icon", "sort", "rating", "filter"]:
                base_shade = 189
            else:
                base_shade = 224
            # print(label)
            label_image, fw = element_resize(label_image, w, h, label_resize, base_shade)
            if label == "image" or label == "icon":
                cv2.line(label_image, (0, 0), (w - 1, h - 1), (79, 79, 79), thickness=1)
                cv2.line(label_image, (0, h - 1), (w - 1, 0), (79, 79, 79), thickness=1)
            if label_resize == 4:
                fw = 0
            if label_text is not None:
                text = label_text[0]
                if label in element_counted.keys():
                    c = element_counted[label]
                    if len(label_text) > c:
                        text = label_text[c]
                label_image, text_ignored = add_text_pil(label_image, text, 0, label_resize, fw)
                if text_ignored and label_resize == 3:
                    label_image, fw = element_resize(base_label_image, w, h, flag=2, base_shade=189)
            try:
                base_image[y : y + h, x : x + w, :] = label_image
                if label in element_counted.keys():
                    element_counted[label] += 1
                else:
                    element_counted[label] = 1
            except Exception as e:
                print(e)
    # base_image = cv2.rectangle(base_image, (0,0), (img_w-1, img_h-1), (0,0,0), thickness=2)
    cv2.imwrite(dst_file_path, base_image)



from src.semanticJsonParser import SemanticJsonParser
from src.uiLabelFileManager import UILabelFileManager

def get_elements(path, real):
    elements = []
    try:
        data = None
        with open(path, "r") as f:
            data = json.load(f)
            if real:
                return SemanticJsonParser.read_json(data, label_hierarchy_map)
            else:
                shapes = data["shapes"]
                flags = data["flags"]
                for shape in shapes:
                    label = shape["label"]
                    points = shape["points"]
                    elements.append(
                        [label, [int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1])]]
                    )
    except Exception as e:
        print(e)
    return elements





def create_json_file(path, elements, flag):
    data = {"shapes": elements,
            "imageHeight": 567,
            "imageWidth": 360,
            "flags": {flag: True}
            }
    if data is not None and len(data) > 0:
        with open(path, "w+") as ff:
            json.dump(data, ff, indent=True)


def get_category_value(category: UIDesignPattern):
    if category == UIDesignPattern.login:
        return 0
    elif category == UIDesignPattern.account_creation:
        return 1
    elif category == UIDesignPattern.product_listing:
        return 2
    elif category == UIDesignPattern.product_description:
        return 3
    else:
        return 4


def generate_wireframe_samples(category: UIDesignPattern, sample_num=16, z_dim=128):
    global GEN

    z = tf.random.truncated_normal(shape=(sample_num, z_dim), dtype=tf.float32)
    print(z.shape,z)
    c = tf.ones(sample_num, dtype=tf.int32) * get_category_value(category)
    print("c first shape",c.shape,c)
    c = tf.reshape(c, [sample_num, 1])
    print("c second shape",c.shape,c)
    samples = GEN([z, c])[0].numpy()
    print("samples shape",samples.shape)
    images = np.array([resize_screen(x, cv2.INTER_NEAREST) for x in samples])
    
    
    ###
    #  alex added 
    ###


    print(images.shape)
    print(type(images))

    data = im.fromarray(images[0])   
    # saving the final output 
    
    now = datetime.now()
    # convert from datetime to timestamp
    ts = int(datetime.timestamp(now))
    
    # as a PNG file
    file_name='out_api/{}_{}.png'.format(ts,category)
    data.save(file_name)

    # convert to new semantic iamge and annotate it with rectangles   ( post processing)
    # 
    get_bounding_boxes('.', file_name, '.', '.')

    
    # convert semantic image to wireframe


    


    #print(images[0])
    wireframes = [get_wireframe(i, image, category) for i, image in enumerate(images)]
    print(wireframes[7])
    return wireframes
