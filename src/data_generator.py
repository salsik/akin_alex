import os

import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

class DataGenerator():
    def __init__(self, image_paths, labels, image_size, batch_size):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.batch_size = batch_size
        self.generator = self._generator()
        self.dataset = self.get_semantic_images_for_gan_train(batch_size, True)

    def read_image(self, image_path, crop=False):
        image = cv2.imread(image_path)

        if crop:
            image = self.crop_center(image)

        image = cv2.resize(image, (self.image_size, self.image_size))
        return image[:, :, ::-1].astype(np.float32) # convert to RGB and float

    def crop_center(self, image):
        h_center, w_center, shift = image.shape[0] // 2, image.shape[1] // 2, self.image_size//2
        
        return image[
            int(h_center-(shift)):int(h_center-(shift)+self.image_size), 
            int(w_center-(shift)):int(w_center-(shift)+self.image_size)
        ]

    def _generator(self):
        data = np.empty((self.batch_size, self.image_size, self.image_size, 3))
        idxes = np.arange(len(self.image_paths))
        while True:
            np.random.shuffle(idxes)
            
            i = 0   
            while (i + 1) * self.batch_size <= len(self.image_paths):
                batch_paths = self.image_paths[i * self.batch_size:(i + 1) * self.batch_size]

                for j, path in enumerate(batch_paths):
                    img = self.read_image(path, crop=False)
                    data[j] = (img / 127.) - 1
                i += 1
                yield data.astype(np.float32)

    def get_semantic_images_for_gan_train(self, batch_size, cache=None):
        #breakpoint here
        dataset_len = len(self.image_paths)

        list_ds = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels))

        train_dataset = list_ds.map(self.process_path_for_gan, num_parallel_calls=AUTOTUNE)

        if dataset_len % batch_size != 0:
            dataset_len = int(dataset_len / batch_size) * batch_size

        train_dataset = train_dataset.take(dataset_len)

        train_dataset = train_dataset.shuffle(buffer_size=dataset_len).batch(batch_size=self.batch_size)
        # Repeat forever

        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                train_dataset = train_dataset.cache(cache)
            else:
                train_dataset = train_dataset.cache()

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

        return train_dataset

    def process_path_for_gan(self, image_path, label):
        # load the raw data from the file as a string
        img = tf.io.read_file(image_path)
        img = self.decode_img(img, self.image_size, self.image_size, convert_to_bgr=False)

        return img, [label - 1]

    def decode_img(self, img, img_width, img_height, convert_to_bgr=False):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # convert channels to bgr?
        if convert_to_bgr:
            img = img[..., ::-1]
        # resize the image to the desired size.
        img = (img * 2) - 1.0
       # return img
        return tf.image.resize(img, [img_width, img_height])



def plt_display(image, title):
  fig = plt.figure()
  a = fig.add_subplot(1, 1, 1)
  imgplot = plt.imshow(image)
  a.set_title(title)


def get_label_map(path, exclude_categories=["0"], image_extention_flag=True):
    new_data = {}
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = f.readlines()
                for row in data:
                    row = row.replace("\n", "")
                    c = row.split(",")
                    if c[1] not in exclude_categories:
                        if image_extention_flag:
                            new_data[c[0]] = int(c[1])
                        else:
                            new_data[c[0][:-4]] = int(c[1])
        else:
            print("file " + str(path) + " does not exist")
    except Exception as e:
        print(e)
    return new_data

def get_data_generator(category_file,data_path,image_size=128, batch_size=100):

    # label_map is a dictionary to map each image with its corresponding category
    label_map = get_label_map(category_file)
    images = []
    labels = []
    cat_count = {}
    for dirname, dirnames, filenames in os.walk(data_path):
        for f in filenames:
            # the way to solve the problem is to delete all json files from the directory
            if (f== '12514.json'):
                print("stop")
            image_path = os.path.join(dirname, f)
            if (f in label_map):
                image_label = label_map[f]
                
            else:
                image_label= -1
            if image_label in cat_count.keys() and cat_count[image_label] >= 100:
                continue
            else:
                images.append(image_path)
                labels.append(image_label)
                if image_label not in cat_count.keys():
                    cat_count[image_label] = 1
                else:
                    cat_count[image_label] += 1

    print(len(images))
    nbatch = int(np.ceil(len(images) / batch_size))
    returner = DataGenerator(images, labels, image_size=image_size, batch_size=batch_size)
    return returner




def testsomeimages():

    print("start")

    img_path="./data/train/Akin_SAGAN_500/images/product_description/10691.jpg"

    img = cv2.imread(img_path)

   # cv2.imshow("before segmentation",img)
    plt_display(img, 'before segm')

    images=[img_path,img_path,img_path]
    labels=[3,3,3]
    # should comment the line of self.dataset  in constructor
    dg = DataGenerator(images, labels, 128, 16)

    im,lb = dg.process_path_for_gan(img_path,3)

    
    #with tf.compat.v1.Session() as sess:
       # tf.compat.v1.disable_eager_execution()
    image_tf = im#.numpy()
    
    print(image_tf.dtype)
    plt_display(image_tf, 'TF')

    
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def testDb():
    print ("test db")

    data_generator = get_data_generator("./resources/category.csv","/data1/data_alex/rico/Akin_SAGAN_500/semantic",128,50) # ./resources/category.csv'

    count= 0
    

    for image_batch in data_generator.dataset:

        print(image_batch[0].shape)
        print(image_batch[1])
            
        # just to sisplay the image  afterdecoding and resizing
        #plt_display(image_batch[0][0],"dsds")
        count+=1
        break


        
    
    print("count",count)


if __name__ == "__main__":

    # test 1
    #testsomeimages()
    testDb()
