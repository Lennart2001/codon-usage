from math import nan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import tensorflow as tf


def get_table_row(model, macro, micro, weighted, acc, loss, auc) -> str:
    temp_str = "|     "
    temp_str += str(model)
    if len(model) == 3:
        temp_str += " "
    temp_str += "   "
    temp_str += f"|  {macro:.6f}  |  {micro:.6f}  |  {weighted:.6f}  |  {acc:.6f}  |  {loss:.6f}  |  {auc:.6f}  |"
    return temp_str


title_string = "|            |  Macro F1  |  Micro F1  |  Weight F1 |  Accuracy  |    Loss    |     AUC    |"
big_string = ""

datasets = ["https://raw.githubusercontent.com/Lennart2001/codon-usage/main/Datasets/codon_usage_dropped.csv",
            "https://raw.githubusercontent.com/Lennart2001/codon-usage/main/Datasets/codon_usage_median.csv",
            "https://raw.githubusercontent.com/Lennart2001/codon-usage/main/Datasets/codon_usage_mean.csv"]

datasets_names = ["Dropped", "Median", "Mean"]
kingdom_dna = ["Kingdom", "DNAtype"]
optimizers = ["nAdam"]
mode_description = ["AUTO", "GRU", "LSTM", "FNN", "CNN"]

model_num = len(mode_description)


info_string = ""
info_string += "\n * * * * * *    INFO    * * * * * *\n\n"
info_string += "The number of epochs will determine how many training iterations the model will go through per run.\n\n"
info_string += "The number of runs will determine how many times you actually train the model. The results of each run "
info_string += "will be averaged at the end and displayed in a pseudo CSV table.\n\n"
info_string += "The question about updates just means if you want to know where you currently are in the total "
info_string += "training process. It will tell you how many models are left, how many runs are left and which dataset "
info_string += "you are in.\n\n\n * * * * * *    INFO    * * * * * *\n"

print(info_string)

print(mode_description)
print("Current number of Models: " + str(model_num) + "\n")

epochs = int(input("How many epochs for each model? "))
if epochs < 1:
    epochs = 1
runs = int(input("How many runs for each model? "))
if runs < 1:
    runs = 1
updates = str(input("Do you want progress updates? y/n  "))
want_updates = False
if updates in ["yes", "y", "Yes", "YES", "Y", "YEs"]:
    want_updates = True

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

        big_string += title_string + "\n"

        for mod in range(model_num):  # CHANGE FOR AMOUNT OF MODELS USED

            loss_ = 0.0
            accuracy_ = 0.0
            micro_f1 = 0.0
            macro_f1 = 0.0
            weighted_f1 = 0.0
            auc_roc = 0.0

            for epoch in range(runs):  # CHANGE FOR AMOUNT OF TIMES MODEL + OPTIMIZER SHOULD BE RUN

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
                model.compile(optimizer="nadam", loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])

                history = model.fit(train_x, train_y, epochs=epochs, validation_split=0.2, verbose=0)

                loss, acc = model.evaluate(test_x, test_y, verbose=0)

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

                if want_updates:
                    print("Kingdom vs DNA: " + kingdom_dna[kd])
                    print(datasets_names[dataset] + " " + str(dataset + 1) + "/3")
                    print("Model:     " + str(mod + 1) + "/" + str(model_num))
                    print("Runs:      " + str(epoch + 1) + "/" + str(runs))

            loss_ /= runs
            accuracy_ /= runs
            macro_f1 /= runs
            micro_f1 /= runs
            weighted_f1 /= runs
            auc_roc /= runs

            big_string += get_table_row(mode_description[mod], macro_f1, micro_f1, weighted_f1, accuracy_, loss_,
                                        auc_roc) + "\n"

        if kingdom_dna[kd] == "Kingdom":
            print("\n                         Kingdom Classification - " + datasets_names[dataset])
        else:
            print("\n                        DNA Type Classification - " + datasets_names[dataset])

        print(big_string)
        big_string = ""

print("\n\nDATA COLLECTION FINISHED!!!\n\n")
