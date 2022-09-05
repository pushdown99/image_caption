#
# CNN Transformer in PyTorch and TensorFlow 2 w/ Keras
# tf2/ImageCaption/__main__.py
# Copyright 2022 Haeyeon, Hwang
#
# Main module for the TensorFlow/Keras implementation of Image Captioninh. Run this
# from the root directory, e.g.:
#
# python -m tf2.ImageCaption --help
#

#
# TODO
# ----
# - 
#

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print('')
print('________                               _______________')
print('___  __/__________________________________  ____/__  /________      __')
print('__  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /')
print('_  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ /')
print('/_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/')
print('')


import argparse
import numpy as np
import random
from tqdm import tqdm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

#from .statistics import TrainingStatistics
#from .statistics import PrecisionRecallCurveCalculator
from .datasets  import coco2014
from .models    import caption
#from .models import vgg16
#from .models import math_utils
#from .models import anchors
#from . import utils
#from . import visualize

if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--train", action = "store_true", help = "Train model")
  group.add_argument("--eval", action = "store_true", help = "Evaluate model")
  group.add_argument("--predict", metavar = "url", action = "store", type = str, help = "Run inference on image and display detected boxes")
  group.add_argument("--predict-to-file", metavar = "url", action = "store", type = str, help = "Run inference on image and render detected boxes to 'predictions.png'")
  group.add_argument("--predict-all", metavar = "name", action = "store", type = str, help = "Run inference on all images in the specified dataset split and write to directory 'predictions_${split}'")
  parser.add_argument("--load-from", metavar = "file", action = "store", help = "Load initial model weights from file")
  parser.add_argument("--save-to", metavar = "file", action = "store", help = "Save final trained weights to file")
  parser.add_argument("--save-best-to", metavar = "file", action = "store", help = "Save best weights (highest mean average precision) to file")
  parser.add_argument("--dataset-dir", metavar = "dir", action = "store", default = "VOCdevkit/VOC2007", help = "VOC dataset directory")
  parser.add_argument("--train-split", metavar = "name", action = "store", default = "trainval", help = "Dataset split to use for training")
  parser.add_argument("--eval-split", metavar = "name", action = "store", default = "test", help = "Dataset split to use for evaluation")
  parser.add_argument("--cache-images", action = "store_true", help = "Cache images during training (requires ample CPU memory)")
  parser.add_argument("--periodic-eval-samples", metavar = "count", action = "store", default = 1000, help = "Number of samples to use during evaluation after each epoch")
  parser.add_argument("--checkpoint-dir", metavar = "dir", action = "store", help = "Save checkpoints after each epoch to the given directory")
  parser.add_argument("--plot", action = "store_true", help = "Plots the average precision of each class after evaluation (use with --train or --eval)")
  parser.add_argument("--log-csv", metavar = "file", action = "store", help = "Log training metrics to CSV file")
  parser.add_argument("--epochs", metavar = "count", type = int, action = "store", default = 1, help = "Number of epochs to train for")
  parser.add_argument("--optimizer", metavar = "name", type = str, action = "store", default = "sgd", help = "Optimizer to use (\"sgd\" or \"adam\")")
  parser.add_argument("--learning-rate", metavar = "value", type = float, action = "store", default = 1e-3, help = "Learning rate")
  parser.add_argument("--clipnorm", metavar = "value", type = float, action = "store", default = 0.0, help = "Gradient norm clipping (use 0 for none)")
  parser.add_argument("--momentum", metavar = "value", type = float, action = "store", default = 0.9, help = "SGD momentum")
  parser.add_argument("--beta1", metavar = "value", type = float, action = "store", default = 0.9, help = "Adam beta1 parameter (decay rate for 1st moment estimates)")
  parser.add_argument("--beta2", metavar = "value", type = float, action = "store", default = 0.999, help = "Adam beta2 parameter (decay rate for 2nd moment estimates)")
  parser.add_argument("--weight-decay", metavar = "value", type = float, action = "store", default = 5e-4, help = "Weight decay")
  parser.add_argument("--dropout", metavar = "probability", type = float, action = "store", default = 0.0, help = "Dropout probability after each of the two fully-connected detector layers")
  parser.add_argument("--custom-roi-pool", action = "store_true", help = "Use custom RoI pool implementation instead of TensorFlow crop-and-resize with max-pool (much slower)")
  parser.add_argument("--detector-logits", action = "store_true", help = "Do not apply softmax to detector class output and compute loss from logits directly")
  parser.add_argument("--no-augment", action = "store_true", help = "Disable image augmentation (random horizontal flips) during training")
  parser.add_argument("--exclude-edge-proposals", action = "store_true", help = "Exclude proposals generated at anchors spanning image edges from being passed to detector stage")
  parser.add_argument("--dump-anchors", metavar = "dir", action = "store", help = "Render out all object anchors and ground truth boxes from the training set to a directory")
  parser.add_argument("--debug-dir", metavar = "dir", action = "store", help = "Enable full TensorFlow Debugger V2 logging to specified directory")
  options = parser.parse_args()

  # Run-time environment
  cuda_available = tf.test.is_built_with_cuda()
  #gpu_available = tf.test.is_gpu_available(cuda_only = False, min_cuda_compute_capability = None)
  gpu_available = len(tf.config.list_physical_devices('GPU'))

  print("CUDA Available : %s" % ("yes" if cuda_available else "no"))
  print("GPU Available  : %s" % ("yes" if gpu_available else "no"))
  print("Eager Execution: %s" % ("yes" if tf.executing_eagerly() else "no"))

  dataset = coco2014.Dataset()
  train_dataset, valid_dataset = dataset.Load()

  model = caption.Model(dataset.get_vocab_size())
  model.Dataset (train_dataset, valid_dataset)
  model.Compile ()
  model.Fit ()

  # Compute definitive metrics on train/valid set
  train_metrics = model.evaluate(train_dataset, batch_size=BATCH_SIZE)
  valid_metrics = model.evaluate(valid_dataset, batch_size=BATCH_SIZE)

