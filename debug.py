import copy
import os

import numpy as np
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from keras.models import load_model
import json
from time import *
import argparse

from grad_cam import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--d", "-d", help="Dataset", type=str, default="cifar10")
parser.add_argument("--m", "-m", help="Model", type=str, default="conv")
parser.add_argument("--scale", "-scale", help="The scale of heatmap", type=float, default=0.85)
parser.add_argument("--save_path", "-save_path", help="Save path", type=str, default="./tmp/")
parser.add_argument("--batch_size", "-batch_size", help="Batch size", type=int, default=128)
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


def layers_heatmap(input_image, classes_obs, select, selected_layers, wrong_idx):
    layer_heatmap = []
    for layer_idx in range(len(layer_names)):
        if select and (layer_idx not in selected_layers):
            continue

        observed_class = int(classes_obs[wrong_idx][layer_idx])

        heatmap = grad_cam(grad_layer_class, input_image, observed_class, layer_idx)
        layer_heatmap.append(heatmap)

    return layer_heatmap


def cal_heatmap_diff(input_image, layer_heatmap, select, wrong_idx, classes_obs, selected_layers, args):
    if select:
        observed_class = np.array(list(map(int, classes_obs[wrong_idx])))[selected_layers]
    else:
        observed_class = np.array(list(map(int, classes_obs[wrong_idx])))

    heatmap_diff = np.zeros([input_image.shape[1], input_image.shape[2]])
    count = 0
    for label in range(args.num_classes):
        if label in observed_class:
            label_idxes = np.where(observed_class == label)[0]
            num_idxes = label_idxes.shape[0]
            heatmap_diff += np.sum(np.array(layer_heatmap)[label_idxes], axis=0) / num_idxes
            count += 1
    heatmap_diff /= count

    return heatmap_diff


def tackle_one_input(wrong_idx, true_class, classes_obs, args):
    predicted_class = test_pred[wrong_idx]
    # print("{} is predicted as {}".format(true_class, predicted_class))

    if select:
        selected_layers = layers_agree[str(predicted_class)]
        if selected_layers == []:
            selected_layers = [x for x in range(num_conv_layers)]
    else:
        selected_layers = None
    # print("selected layers: {}".format(selected_layers))

    input_image = x_test[wrong_idx]
    input_image = np.expand_dims(input_image, axis=0)

    layer_heatmap = layers_heatmap(input_image, classes_obs, select, selected_layers, wrong_idx)
    heatmap_diff = cal_heatmap_diff(input_image, layer_heatmap, select, wrong_idx, classes_obs, selected_layers, args)
    heatmap_cam = grad_cam(grad_layer_class, input_image, predicted_class, (num_conv_layers-1))

    return heatmap_diff, heatmap_cam


if __name__ == "__main__":

    # layer names
    if args.m == "conv":
        num_conv_layers = 6
        layer_names = []
        if args.d == "fmnist":
            num_layers = 8
        else:
            num_layers = 9
        for i in range(1, num_layers+1):
            layer_names.append("activation_" + str(i))
    elif args.m == "vgg16":
        num_conv_layers = 13
        layer_names = []
        num_layers = 15
        for i in range(1, num_layers+1):
            layer_names.append("activation_" + str(i))
    else:
        num_conv_layers = 19
        layer_names = []
        num_layers = 20
        for i in range(1, num_layers):
            layer_names.append("activation_" + str(i))
        layer_names.append("dense_1")

    # load dataset and models
    x_train_total = x_test = y_train_total = y_test = None
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

    test_prob = orig_model.predict(x_test)
    test_pred = np.argmax(test_prob, axis=1)
    model_accuracy = np.mean(test_pred == y_test)
    print("model accuracy: {}".format(model_accuracy))
    pred_labels = np.load("./tmp/" + path_name + "/pred_labels_test.npy")
    # read selected layers by kde
    filename = "./tmp/" + path_name + "/selected_layers_agree.json"
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            layers_agree = json.load(json_file)
        json_file.close()
    else:
        layers_agree = {}
        for label in range(args.num_classes):
            filename = "./tmp/" + path_name + "/selected_layers_agree_" + str(label) + ".json"
            with open(filename, "r") as json_file:
                layers_agree_label = json.load(json_file)
            json_file.close()
            layers_agree.update(layers_agree_label)

    for i in range(len(layers_agree)):
        layers = copy.deepcopy(layers_agree[str(i)])
        for j in range(len(layers_agree[str(i)])):
            if layers_agree[str(i)][j] >= num_conv_layers:
                layers.remove(layers_agree[str(i)][j])
        layers_agree[str(i)] = layers
    TP_idx = np.load("./tmp/" + path_name + "/TP_idx.npy")
    FP_idx = np.load("./tmp/" + path_name + "/FP_idx.npy")
    wrong_idxes = np.where(pred_labels.T[-1] != pred_labels.T[-2])[0]

    grad_layer_class = store_grad_for_layer_class(orig_model, layer_names, args.num_classes)

    select = True
    scale_heatmap_diff = args.scale

    classes_obs = pred_labels[:, :num_conv_layers]

    idx_wrong = []
    input_wrong_m = []
    num_mask_wrong = []
    mask_wrong = []

    begin_time = time()
    for idx in wrong_idxes:
        idx_wrong.append(idx)

        input_wrong = x_test[idx]
        true_class = y_test[idx]

        heatmap_diff, heatmap_cam = tackle_one_input(idx, true_class, classes_obs, args)
        input_wrong = x_test[idx]
        mask = np.ones_like(heatmap_diff)

        mask[heatmap_diff > scale_heatmap_diff * np.max(heatmap_diff)] = 0.0

        num_mask = np.where(mask == 0.0)[0].shape[0]
        num_mask_wrong.append(num_mask)

        mask = np.expand_dims(mask, axis=-1).repeat(input_wrong.shape[-1], axis=-1)
        mask_wrong.append(mask)

        input_tmp = copy.deepcopy(input_wrong)
        input_tmp[np.where(mask == 0.0)] = -0.5
        input_wrong_m.append(input_tmp)

    mask_wrong = np.array(mask_wrong)

    end_time = time()
    run_time = end_time - begin_time
    print('running time: ', run_time / mask_wrong.shape[0])

    idx_wrong = np.array(idx_wrong)
    input_wrong_m = np.array(input_wrong_m)
    print("mean of mask: {}, min: {}, max: {}".format(np.mean(num_mask_wrong), np.min(num_mask_wrong), np.max(num_mask_wrong)))

    prob_mask = orig_model.predict(input_wrong_m)

    pred_mask = np.argmax(prob_mask, axis=1)

    true_labels = y_test[idx_wrong]
    print("acc of masked: {} {}".format(np.sum(pred_mask == true_labels), np.sum(pred_mask == true_labels) / true_labels.shape[0]))

    pred_labels_orig = test_pred[idx_wrong]
    print("change of masked: {}".format(np.sum(pred_mask != pred_labels_orig)))

    mask_path = "./tmp/" + path_name + "/repair/" + str(scale_heatmap_diff) + "/"
    dir = os.path.dirname(mask_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(mask_path + "input_wrong.npy", np.uint8((input_wrong_m + 0.5) * 255.0))
    np.save(mask_path + "mask_wrong.npy", (1.0 - mask_wrong))



    idx_wrong = []
    input_wrong_m = []
    num_mask_wrong = []
    mask_wrong = []

    for idx in TP_idx:
        idx_wrong.append(idx)

        input_wrong = x_test[idx]
        true_class = y_test[idx]

        heatmap_diff, heatmap_cam = tackle_one_input(idx, true_class, classes_obs, args)
        input_wrong = x_test[idx]
        mask = np.ones_like(heatmap_diff)

        mask[heatmap_diff > scale_heatmap_diff * np.max(heatmap_diff)] = 0.0

        num_mask = np.where(mask == 0.0)[0].shape[0]
        num_mask_wrong.append(num_mask)

        mask = np.expand_dims(mask, axis=-1).repeat(input_wrong.shape[-1], axis=-1)
        mask_wrong.append(mask)

        input_tmp = copy.deepcopy(input_wrong)
        input_tmp[np.where(mask == 0.0)] = -0.5
        input_wrong_m.append(input_tmp)

    mask_wrong = np.array(mask_wrong)

    idx_wrong = np.array(idx_wrong)
    input_wrong_m = np.array(input_wrong_m)
    print("mean of mask: {}, min: {}, max: {}".format(np.mean(num_mask_wrong), np.min(num_mask_wrong), np.max(num_mask_wrong)))

    prob_mask = orig_model.predict(input_wrong_m)

    pred_mask = np.argmax(prob_mask, axis=1)

    true_labels = y_test[idx_wrong]
    print("acc of masked: {} {}".format(np.sum(pred_mask == true_labels), np.sum(pred_mask == true_labels) / true_labels.shape[0]))

    pred_labels_orig = test_pred[idx_wrong]
    print("change of masked: {}".format(np.sum(pred_mask != pred_labels_orig)))

    mask_path = "./tmp/" + path_name + "/repair/" + str(scale_heatmap_diff) + "/"
    dir = os.path.dirname(mask_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(mask_path + "input_TP.npy", np.uint8((input_wrong_m + 0.5) * 255.0))
    np.save(mask_path + "mask_TP.npy", (1.0 - mask_wrong))


    idx_wrong = []
    input_wrong_m = []
    num_mask_wrong = []
    mask_wrong = []

    for idx in FP_idx:
        idx_wrong.append(idx)

        input_wrong = x_test[idx]
        true_class = y_test[idx]

        heatmap_diff, heatmap_cam = tackle_one_input(idx, true_class, classes_obs, args)
        input_wrong = x_test[idx]
        mask = np.ones_like(heatmap_diff)

        mask[heatmap_diff > scale_heatmap_diff * np.max(heatmap_diff)] = 0.0

        num_mask = np.where(mask == 0.0)[0].shape[0]
        num_mask_wrong.append(num_mask)

        mask = np.expand_dims(mask, axis=-1).repeat(input_wrong.shape[-1], axis=-1)
        mask_wrong.append(mask)

        input_tmp = copy.deepcopy(input_wrong)
        input_tmp[np.where(mask == 0.0)] = -0.5
        input_wrong_m.append(input_tmp)

    mask_wrong = np.array(mask_wrong)

    idx_wrong = np.array(idx_wrong)
    input_wrong_m = np.array(input_wrong_m)
    print("mean of mask: {}, min: {}, max: {}".format(np.mean(num_mask_wrong), np.min(num_mask_wrong), np.max(num_mask_wrong)))

    prob_mask = orig_model.predict(input_wrong_m)

    pred_mask = np.argmax(prob_mask, axis=1)

    true_labels = y_test[idx_wrong]
    print("acc of masked: {} {}".format(np.sum(pred_mask == true_labels), np.sum(pred_mask == true_labels) / true_labels.shape[0]))

    pred_labels_orig = test_pred[idx_wrong]
    print("change of masked: {}".format(np.sum(pred_mask != pred_labels_orig)))

    mask_path = "./tmp/" + path_name + "/repair/" + str(scale_heatmap_diff) + "/"
    dir = os.path.dirname(mask_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(mask_path + "input_FP.npy", np.uint8((input_wrong_m + 0.5) * 255.0))
    np.save(mask_path + "mask_FP.npy", (1.0 - mask_wrong))