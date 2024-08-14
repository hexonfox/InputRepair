import tensorflow as tf
from tensorflow.keras import layers, losses, utils
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import os
import numpy as np
from tqdm import tqdm
import argparse
from kdes_generation import fetch_kdes_tf, _get_train_ats
from utils import *
import gc


parser = argparse.ArgumentParser()
parser.add_argument("--d", "-d", help="Dataset", type=str, default="cifar10")
parser.add_argument("--m", "-m", help="Model", type=str, default="conv")
parser.add_argument("--save_path", "-save_path", help="Save path", type=str, default="./tmp/")
parser.add_argument("--batch_size", "-batch_size", help="Batch size", type=int, default=16)
parser.add_argument("--var_threshold", "-var_threshold", help="Variance threshold", type=float, default=1e-5)
parser.add_argument("--eps", "-eps", help="learning step", type=float, default=0.001)
parser.add_argument("--num_classes", "-num_classes", help="The number of classes", type=int, default=10)
parser.add_argument("--gpu", type=str, default='0')


args = parser.parse_args()
args.save_path = args.save_path + args.d + "/" + args.m + "/"
dir = os.path.dirname(args.save_path)
if not os.path.exists(dir):
    os.makedirs(dir)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # set GPU Limits
for d in tf.config.list_physical_devices('GPU'): 
    tf.config.experimental.set_memory_growth(d, True) 
print(args)


def repair(idxes, idx_name, eps, mask=None):
    _, _ = _get_train_ats(orig_model, x_train, layer_names, args)
    gen_img = x_test[idxes]
    label = y_test[idxes]
    count0 = np.sum(pred_labels_test[idxes][:, :-2] == np.expand_dims(pred_labels_test[idxes][:, -2], axis=1), axis=1)
    count = False

    for _ in range(5):
        with tf.GradientTape(persistent=True) as g:
            gen_img = tf.Variable(gen_img)
            pred_layer, pred = fetch_kdes_tf(orig_model, x_train, gen_img, y_train, layer_names, "gen_TP", args)
            pred_layer_arg = tf.argmax(pred_layer, axis=-1).numpy()
            pred_arg = tf.argmax(pred, axis=-1).numpy()

            loss_list = []
            for ix, pred_layer_arg_x in enumerate(tf.transpose(pred_layer_arg)):
                if np.where(pred_layer_arg_x == pred_arg[ix])[0].shape[0] == 0:
                    selected_layers = [-1]
                else:
                    selected_layers = np.where(pred_layer_arg_x == pred_arg[ix])[0]

                loss = 0
                for layer_idx in selected_layers:
                    loss += losses.categorical_crossentropy(pred[ix], pred_layer[layer_idx][ix])

                loss_list.append(loss)

            grads = g.gradient(loss_list, gen_img)
            g.watch(grads)
            # print("grads max: {}".format(np.max(grads.numpy())))
        del g
        gc.collect()

        if mask is None:
            gen_img = gen_img + eps * grads
        else:
            gen_img = gen_img + eps * grads * mask
        gen_img = tf.clip_by_value(gen_img, clip_value_min=-0.5, clip_value_max=0.5)
        gen_img = tf.cast(gen_img, tf.float32)

        count1 = np.sum(pred_layer_arg == pred_arg, axis=0)
        count1 = (count1 > count0) & (pred_arg == label)
        count = count | count1

    true_count = np.sum(count)

    del pred_layer, pred, label, gen_img, loss, grads, pred_layer_arg, pred_arg
    gc.collect()

    return true_count


if __name__ == "__main__":

    # layer names
    if args.m == "conv":
        layer_names = []
        if args.d == "fmnist":
            num_layers = 8
        else:
            num_layers = 9
        for i in range(1, num_layers+1):
            layer_names.append("activation_" + str(i))
    elif args.m == "vgg16":
        layer_names = []
        num_layers = 15
        for i in range(1, num_layers+1):
            layer_names.append("activation_" + str(i))
    else:
        layer_names = []
        num_layers = 20
        for i in range(1, num_layers):
            layer_names.append("activation_" + str(i))
        layer_names.append("dense_1")

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

    # data pre-processing
    CLIP_MAX = 0.5
    x_train_total = x_train_total.astype("float32")
    x_train_total = (x_train_total / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    # split original training dataset into training and validation dataset
    x_train = x_train_total[:num_train]
    x_valid = x_train_total[num_train:]
    y_train = y_train_total[:num_train]
    y_valid = y_train_total[num_train:]

    preds_test = np.argmax(orig_model.predict(x_test), axis=1)
    print("original accuracy: {}".format(np.mean(preds_test == y_test)))

    pred_labels_valid = np.load("./tmp/" + path_name + "/pred_labels_valid.npy")
    pred_labels_test = np.load("./tmp/" + path_name + "/pred_labels_test.npy")
    
    TP_idx = np.load("./tmp/" + path_name + "/TP_idx.npy")
    FP_idx = np.load("./tmp/" + path_name + "/FP_idx.npy")
    wrong_idx = np.where(pred_labels_test.T[-1] != pred_labels_test.T[-2])[0]

    eps = args.eps

    print("*****************************wrong*****************************")
    mask = np.load("./tmp/" + path_name + "/repair/0.85/mask_wrong.npy")
    true_count_wrong = repair(wrong_idx, "wrong", eps)
    print("correct: {} in wrong: {}".format(true_count_wrong, wrong_idx.shape[0]))

    print("*****************************TP*****************************")
    mask = np.load("./tmp/" + path_name + "/repair/0.85/mask_TP.npy")
    true_count_TP = repair(TP_idx, "TP", eps)
    print("correct: {} in TP: {}".format(true_count_TP, TP_idx.shape[0]))

    print("*****************************FP*****************************")
    mask = np.load("./tmp/" + path_name + "/repair/0.85/mask_FP.npy")
    true_count_FP = repair(FP_idx, "FP", eps)
    print("correct: {} in FP: {}".format(true_count_FP, FP_idx.shape[0]))

    print("original accuracy: {} vs modified accuracy: {}".format((x_test.shape[0] - wrong_idx.shape[0]) * 100 / x_test.shape[0], (
                    true_count_TP + true_count_FP + x_test.shape[0] - wrong_idx.shape[0] - FP_idx.shape[0]) * 100 / x_test.shape[0]))

