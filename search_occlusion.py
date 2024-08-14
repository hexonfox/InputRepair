import tensorflow as tf
from tensorflow.keras import layers, losses, utils
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import os
import numpy as np
import argparse
from utils import *
from time import *
import copy


parser = argparse.ArgumentParser()
parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
parser.add_argument("--m", "-m", help="Model", type=str, default="conv")
parser.add_argument("--gpu", type=str, default='0')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # set GPU Limits
for d in tf.config.list_physical_devices('GPU'): 
    tf.config.experimental.set_memory_growth(d, True) 
print(args)


# load dataset and models
x_train_total = x_test = y_train_total = y_test = model = None
if args.d == "mnist":
    (x_train_total, y_train_total), (x_test, y_test) = mnist.load_data()
    x_train_total = x_train_total.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    num_train = 50000

    print(infog("y_train len:{}".format(len(y_train_total))))
    print(infog("y_test len:{}".format(len(y_test))))
    # Load pre-trained models.
    orig_model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
    orig_model.summary()

if args.d == "fmnist":
    (x_train_total, y_train_total), (x_test, y_test) = fashion_mnist.load_data()
    x_train_total = x_train_total.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    num_train = 50000

    print(infog("y_train len:{}".format(len(y_train_total))))
    print(infog("y_test len:{}".format(len(y_test))))
    # Load pre-trained models.
    orig_model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
    orig_model.summary()

if args.d == "cifar100":
    (x_train_total, y_train_total), (x_test, y_test) = cifar100.load_data()
    num_train = 40000
    y_train_total = y_train_total.reshape([y_train_total.shape[0]])
    y_test = y_test.reshape([y_test.shape[0]])

    print(infog("y_train len:{}".format(len(y_train_total))))
    print(infog("y_test len:{}".format(len(y_test))))

    orig_model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
    orig_model.summary()

if args.d == "cifar100_coarse":
    (x_train_total, y_train_total), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
    num_train = 40000
    y_train_total = y_train_total.reshape([y_train_total.shape[0]])
    y_test = y_test.reshape([y_test.shape[0]])

    print(infog("y_train len:{}".format(len(y_train_total))))
    print(infog("y_test len:{}".format(len(y_test))))

    orig_model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
    orig_model.summary()

elif args.d == "cifar10":
    (x_train_total, y_train_total), (x_test, y_test) = cifar10.load_data()
    num_train = 40000
    y_train_total = y_train_total.reshape([y_train_total.shape[0]])
    y_test = y_test.reshape([y_test.shape[0]])

    print(infog("y_train len:{}".format(len(y_train_total))))
    print(infog("y_test len:{}".format(len(y_test))))

    orig_model = load_model("./models/model_" + args.d + "_" + args.m + ".h5")
    orig_model.summary()

path_name = args.d + "/" + args.m

test_preds = np.argmax(orig_model.predict(x_test.astype("float32") / 255.0 - 0.5), axis=1)

success_index = np.zeros([x_test.shape[0], 2], dtype=int) - 1
success_index[test_preds != y_test] = -2
print(np.where(success_index == -2)[0].shape)
mask_index = {}
mask_size = 5
for i in range(x_test.shape[1]-mask_size+1):
    for j in range(x_test.shape[2]-mask_size+1):
        count_1 = np.sum(success_index.T[0] == -1)
        if count_1 != 0:
            remain_idx = np.where(success_index.T[0] == -1)[0]
            tmp_x = copy.deepcopy(x_test[remain_idx])
            mask = np.zeros([x_test.shape[1], x_test.shape[2]])
            mask[i:(i+5),j:(j+5)] = 1.0
            mask = np.expand_dims(mask, axis=-1).repeat(x_test.shape[-1], axis=-1)
            tmp_x[:,mask == 1.0] = 255
            preds = np.argmax(orig_model.predict(tmp_x.astype("float32") / 255.0 - 0.5), axis=1)
            right_count = np.sum(preds == y_test[remain_idx])
            print("right_count: {}".format(right_count))
            if right_count == count_1:
                continue

            success_index[remain_idx[preds != y_test[remain_idx]]] = [i, j]
            
success_index[test_preds != y_test] = -1

np.save("./tmp/" + path_name + "/success_index.npy", success_index)




