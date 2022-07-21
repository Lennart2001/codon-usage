from math import nan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import tensorflow as tf
import statistics
import matplotlib.pyplot as plt

datasets = ["/home/hacker/Desktop/Codon Usage/Datasets/codon_usage_dropped.csv",
            "/home/hacker/Desktop/Codon Usage/Datasets/codon_usage_median.csv",
            "/home/hacker/Desktop/Codon Usage/Datasets/codon_usage_mean.csv"]

datasets_names = ["Dropped", "Median", "Mean"]
kingdom_dna = ["Kingdom", "DNAtype"]
optimizers = ["nAdam"]


for dataset in range(3):
    for kd in range(2):
        data = pd.read_csv(datasets[dataset])

        if kd == 0:
            t_x = data.drop(["SpeciesID", "Ncodons", "SpeciesName", kingdom_dna[1], "Unnamed: 0"], axis=1)
        else:
            t_x = data.drop(["SpeciesID", "Ncodons", "SpeciesName", kingdom_dna[0], "Unnamed: 0"], axis=1)

        x_train = np.float32(t_x.drop([kingdom_dna[kd]], axis=1))
        y_train = LabelEncoder().fit_transform(t_x[kingdom_dna[kd]])

        train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.2, random_state=7)

        mode_description = ["AUTO", "GRU", "LSTM", "FNN", "CNN"]

        epochs = 5
        model_num = len(mode_description)

        for mod in range(model_num):  # CHANGE FOR AMOUNT OF MODELS USED
            for opt in optimizers:
                print("Using Optimizer: " + opt)

                loss_ = 0.0
                accuracy_ = 0.0
                micro_f1 = 0.0
                macro_f1 = 0.0
                weighted_f1 = 0.0
                auc_roc = 0.0

                for epoch in range(epochs):  # CHANGE FOR AMOUNT OF TIMES MODEL + OPTIMIZER SHOULD BE RUN

                    models = [tf.keras.Sequential([
                        tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),

                        tf.keras.layers.Dense(2, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),

                        tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(11, activation="softmax")
                    ]), tf.keras.Sequential([
                        tf.keras.layers.Reshape((1, 64)),
                        tf.keras.layers.GRU(64, return_sequences=True),
                        tf.keras.layers.GRU(64, return_sequences=True),
                        tf.keras.layers.GRU(64),

                        tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(11, activation="softmax")
                    ]), tf.keras.Sequential([
                        tf.keras.layers.Reshape((1, 64)),
                        tf.keras.layers.LSTM(64, return_sequences=True),
                        tf.keras.layers.LSTM(64, return_sequences=True),
                        tf.keras.layers.LSTM(64),

                        tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(11, activation="softmax")
                    ]), tf.keras.Sequential([

                        tf.keras.layers.Reshape((8, 8, 1)),
                        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(8, 8, 1)),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Conv2D(128, (2, 2), activation="relu"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Conv2D(256, (2, 2), activation="relu"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.MaxPooling2D((2, 2)),

                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(11, activation="softmax")

                    ]), tf.keras.Sequential([
                        tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU()),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(11, activation="softmax")

                    ])]

                    model = models[mod]
                    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                  metrics=["accuracy"])

                    history = model.fit(train_x, train_y, epochs=200, validation_split=0.2, verbose=1)

                    print("\nCurrent Model:")
                    loss, acc = model.evaluate(test_x, test_y, verbose=2)
                    print("Current Model Accuracy: {:5.2f}%".format(100 * acc))
                    print("Current Model Loss:     ", str(loss))
                    print()

                    pred = model.predict(test_x)
                    prediction = np.argmax(pred, axis=1)

                    macro = f1_score(test_y, prediction, average="macro")
                    micro = f1_score(test_y, prediction, average="micro")
                    weighted = f1_score(test_y, prediction, average="weighted")

                    loss_ += loss
                    accuracy_ += acc
                    macro_f1 += macro
                    micro_f1 += micro
                    weighted_f1 += weighted

                    try:
                        roc_auc = roc_auc_score(test_y, pred, multi_class="ovr")
                        auc_roc += roc_auc
                    except ValueError:
                        auc_roc += 0.0

                    print("Kingdom vs DNA: " + kingdom_dna[kd])
                    print("Dataset:   " + str(dataset+1) + "/2")
                    print(opt + ":    " + str(epoch+1) + "/" + str(epochs))
                    print("Model:     " + str(mod+1) + "/" + str(model_num))

                save_path = "/home/hacker/Desktop/Codon Usage/" + kingdom_dna[kd] + "/" + mode_description[mod] + "/" + datasets_names[dataset] + "/"

                loss_ /= epochs
                accuracy_ /= epochs
                macro_f1 /= epochs
                micro_f1 /= epochs
                weighted_f1 /= epochs
                auc_roc /= epochs

                with open(save_path + "eval_" + opt + ".csv", "a") as myFile:
                    myFile.write("Loss,Accuracy,Macro F1,Micro F1,Weighted F1,AUC\n")
                    myFile.write(str(loss_) + "," + str(accuracy_) + "," + str(macro_f1) + "," + str(micro_f1) + "," + str(weighted_f1) + "/" + str(auc_roc) + "\n")

print("\n\nDATA COLLECTION FINISHED!!!\n\n")

