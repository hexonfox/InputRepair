from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import Model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
tf.compat.v1.disable_eager_execution()


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]


def store_grad_for_layer_class(input_model, layer_names, nb_classes=1000):
    grad_layer_class = []
    for layer_name in layer_names:
        grad_class = []
        for category_index in range(nb_classes):
            target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
            x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(input_model.output)
            model = Model(inputs=input_model.input, outputs=x)
            loss = K.sum(model.output)
            # conv_output =  [l for l in model.layers if l.name is layer_name][0].output
            conv_output = model.get_layer(layer_name).output
            grads = normalize(_compute_gradients(loss, [conv_output])[0])
            gradient_function = K.function([model.input], [conv_output, grads])
            grad_class.append(gradient_function)
        grad_layer_class.append(grad_class)
    return grad_layer_class


def grad_cam(grad_layer_class, image, category_index, layer_idx):
    output, grads_val = grad_layer_class[layer_idx][category_index]([image])
    output, grads_val = output[0, :], grads_val[0, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (image.shape[1], image.shape[2]))
    cam = np.maximum(cam, 0)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    heatmap = cam

    return heatmap


def precess_cam(image, heatmap, add_image=False):
    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image = (image + 0.5) * 255.0

    # make image to be high resolution
    img_size = (image.shape[0] * 10, image.shape[1] * 10)
    image_resized = cv2.resize(image.squeeze(), img_size)
    if np.ndim(image_resized) < 3:
        # image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    heatmap_resized = cv2.resize(heatmap, img_size)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    if add_image:
        cam = np.float32(cam) + np.float32(image_resized)
    else:
        cam = np.float32(cam)
    cam = 255 * cam / np.max(cam)

    return np.uint8(cam), heatmap_resized

