from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import os
import numpy as np
import copy


data = ["mnist", "fmnist", "cifar10", "cifar100"]
model = ["conv", "vgg16", "resnet"]
scale = 0.6

for d in data:
    for m in model:
        if d == "mnist":
            (x_train_total, y_train_total), (x_test, y_test) = mnist.load_data()
            x_train_total = x_train_total.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
            num_train = 50000

        if d == "fmnist":
            (x_train_total, y_train_total), (x_test, y_test) = fashion_mnist.load_data()
            x_train_total = x_train_total.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
            num_train = 50000

        if d == "cifar100":
            (x_train_total, y_train_total), (x_test, y_test) = cifar100.load_data()
            num_train = 40000
            y_train_total = y_train_total.reshape([y_train_total.shape[0]])
            y_test = y_test.reshape([y_test.shape[0]])

        elif d == "cifar10":
            (x_train_total, y_train_total), (x_test, y_test) = cifar10.load_data()
            num_train = 40000
            y_train_total = y_train_total.reshape([y_train_total.shape[0]])
            y_test = y_test.reshape([y_test.shape[0]])

        path_name = d + "/" + m
        success_index = np.load("./tmp/" + path_name + "/success_index.npy")
        print(np.where(success_index.T[0] == -1)[0].shape)
        idxes = np.where(success_index.T[0] != -1)[0]
        y_test_bd = y_test[idxes]

        x_test_bd = []
        for i in range(idxes.shape[0]):
            mask = np.zeros([x_test.shape[1], x_test.shape[2]])
            idx_0 = success_index[idxes[i]][0]
            idx_1 = success_index[idxes[i]][1]
            mask[idx_0:(idx_0+5),idx_1:(idx_1+5)] = 1.0
            mask = np.expand_dims(mask, axis=-1).repeat(x_test.shape[-1], axis=-1)
            tmp_x = copy.deepcopy(x_test[idxes[i]])
            tmp_x[mask == 1.0] = 255
            x_test_bd.append(tmp_x)
        x_test_bd = np.array(x_test_bd)

        print(y_test_bd[:10])
        # check accuracy of backdoor dataset
        # orig_model = load_model("./models/model_" + d + "_" + m + ".h5")
        # print(np.mean(np.argmax(orig_model.predict(x_test_bd.astype("float32") / 255.0 - 0.5), axis=1) == y_test_bd))
        np.save("./tmp/" + path_name + "/x_test_bd.npy", x_test_bd)
        np.save("./tmp/" + path_name + "/y_test_bd.npy", y_test_bd)