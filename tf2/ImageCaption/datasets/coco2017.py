import re
import os
import json
import math
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import TextVectorization

from ..config import *

strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
AUTOTUNE = tf.data.AUTOTUNE

class Dataset:
  def __init__(self, dir = "dataset/COCO", augment = True, shuffle = True, reduce = True, cache = True, verbose = True):
    if not os.path.exists(dir):
      raise FileNotFoundError("Dataset directory does not exist: %s" % dir)

    self.dir            = dir
    self.verbose        = verbose
    self.train_json = os.path.join(self.dir, 'annotations/coco2017.train.json')
    self.valid_json = os.path.join(self.dir, 'annotations/coco2017.valid.json')
    self.text_json  = os.path.join(self.dir, 'annotations/coco2017.text.json' )

    if not os.path.exists(self.train_json):
      raise FileNotFoundError("Dataset directory does not exist: %s" % self.train_json)
    if not os.path.exists(self.valid_json):
      raise FileNotFoundError("Dataset directory does not exist: %s" % self.valid_json)
    if not os.path.exists(self.text_json):
      raise FileNotFoundError("Dataset directory does not exist: %s" % self.text_json)

    self.augment    = augment
    self.shuffle    = shuffle
    self.cache      = cache
    self.reduce     = reduce

    self.img_transf = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomContrast(factor=(0.05, 0.15)),
      #image_aug.RandomBrightness(brightness_delta=(-0.15, 0.15)),
      #image_aug.PowerLawTransform(gamma=(0.8,1.2)),
      #image_aug.RandomSaturation(sat=(0, 2)),
      #image_aug.RandomHue(hue=(0, 0.15)),
      #tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
      tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
      tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
      tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.10, 0.10))])

    # Load
    with open(self.train_json) as json_file:
      self.train_data = json.load(json_file)
    with open(self.valid_json) as json_file:
      self.valid_data = json.load(json_file)
    with open(self.text_json) as json_file:
      self.text_data = json.load(json_file)

    if self.verbose:
      print('Original sample: train {:,}, valid {:,}, text {:,}'.format(len(self.train_data), len(self.valid_data), len(self.text_data)))

    if self.reduce:
      self.train_data, self.valid_data = self.reduce_dataset_dim(self.train_data, self.valid_data)
      if self.verbose:
        print('Reduced sample : train {:,}, valid {:,}, text {:,}'.format(len(self.train_data), len(self.valid_data), len(self.text_data)))

    # Define tokenizer of Text Dataset
    self.tokenizer = TextVectorization (
      max_tokens				= MAX_VOCAB_SIZE,
      output_mode				= "int",
      output_sequence_length	= SEQ_LENGTH,
      standardize				= self.custom_standardization,
    )

    # Adapt tokenizer to Text Dataset
    print('Vectorization  : {:,}'.format(len(self.text_data)))
    self.tokenizer.adapt(self.text_data)

    # Define vocabulary size of Dataset
    self.vocab_size = len(self.tokenizer.get_vocabulary())
    print('vocabulary size: {:,}'.format(self.vocab_size))

    # 20k images for validation set and 13432 images for test set
    self.valid_data, self.test_data  = self.valid_test_split(self.valid_data)
    if self.verbose:
      print('Split val/test : sample: (train {:,}, valid {:,}, test {:,}, text {:,})'.format(len(self.train_data), len(self.valid_data), len(self.test_data), len(self.text_data)))

  def get_vocab_size(self):
    return self.vocab_size

  def reduce_dataset_dim(self, captions_mapping_train, captions_mapping_valid):
    train_data = {}
    conta_train = 0
    for id in captions_mapping_train:
      if conta_train<=NUM_TRAIN_IMG:
        train_data.update({id : captions_mapping_train[id]})
        conta_train+=1
      else:
        break

    valid_data = {}
    conta_valid = 0
    for id in captions_mapping_valid:
      if conta_valid<=NUM_VALID_IMG:
        valid_data.update({id : captions_mapping_valid[id]})
        conta_valid+=1
      else:
        break

    return train_data, valid_data

  @tf.keras.utils.register_keras_serializable()
  def custom_standardization(self, input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(strip_chars), '')

  def train_val_split(self, caption_data, train_size=0.8, shuffle=True):
    all_images = list(caption_data.keys())

    if shuffle:
      np.random.shuffle(all_images)

    train_size = int(len(caption_data) * train_size)

    training_data   = { img_name: caption_data[img_name] for img_name in all_images[:train_size] }
    validation_data = { img_name: caption_data[img_name] for img_name in all_images[train_size:] }

    return training_data, validation_data

  def valid_test_split(self, captions_mapping_valid):
    valid_data={}
    test_data={}
    conta_valid = 0
    for id in captions_mapping_valid:
      if conta_valid<NUM_VALID_IMG:
        valid_data.update({id : captions_mapping_valid[id]})
        conta_valid+=1
      else:
        test_data.update({id : captions_mapping_valid[id]})
        conta_valid+=1
    return valid_data, test_data

  def read_image_inf(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

  def read_image(self, data_aug):
    def decode_image(img_path):
      img = tf.io.read_file(img_path)
      img = tf.image.decode_jpeg(img, channels=3)
      img = tf.image.resize(img, IMAGE_SIZE)

      if data_aug:
        img = image_augment(img)

      img = tf.image.convert_image_dtype(img, tf.float32)
      return img

    def image_augment(img):
      img = tf.expand_dims(img, axis=0)
      img = self.img_transf(img)
      img = tf.squeeze(img, axis=0)
      return img

    return decode_image

  def setting_batch_dataset (self, images, captions, toAugment):
    read_image_xx = self.read_image (toAugment)
    img_dataset   = tf.data.Dataset.from_tensor_slices(list(images))
    img_dataset   = (img_dataset.map(read_image_xx, num_parallel_calls=AUTOTUNE))
    cap_dataset   = tf.data.Dataset.from_tensor_slices(captions).map(self.tokenizer, num_parallel_calls=AUTOTUNE)
    dataset       = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset       = dataset.batch(BATCH_SIZE).shuffle(SHUFFLE_DIM).prefetch(AUTOTUNE)
    return dataset

  def Load(self):
    train_dataset = self.setting_batch_dataset (list(self.train_data.keys()), list(self.train_data.values()), toAugment=TRAIN_SET_AUG)
    valid_dataset = self.setting_batch_dataset (list(self.valid_data.keys()), list(self.valid_data.values()), toAugment=VALID_SET_AUG)
    return train_dataset, valid_dataset


