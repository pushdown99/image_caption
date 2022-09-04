#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/__main__.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Main module for the TensorFlow/Keras implementation of Faster R-CNN. Run this
# from the root directory, e.g.:
#
# python -m tf2.ImageCaption --help
#

#
# TODO
# ----
# - Investigate the removal of tf.stop_gradient() from regression loss
#   functions and how to debug its adverse effect on training. How would we
#   have known to use this if we had not noticed it in a different code base?
#

