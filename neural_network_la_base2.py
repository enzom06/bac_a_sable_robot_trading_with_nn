import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# x_entrer = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [4, 1.5]),
#                    dtype=float)  # données d'entrer
# y = np.array(([1], [0], [1], [0], [1], [0], [1], [0]), dtype=float)  # données de sortie /  1 = rouge /  0 = bleu

# Changement de l'échelle de nos valeurs pour être entre 0 et 1
# x_entrer = x_entrer / np.amax(x_entrer, axis=0)  # On divise chaque entré par la valeur max des entrées

# On récupère ce qu'il nous intéresse
# X = np.split(x_entrer, [8])[0]  # Données sur lesquelles on va s'entrainer, les 8 premières de notre matrice
# xPrediction = np.split(x_entrer, [8])[1]  # Valeur que l'on veut trouver


# Notre classe de réseau neuronal
# [nb_in, nb_in_layer, nb_out]


class neural_network(object):
    def __init__(self, list_neuro):

        # Nos paramètres
        self.ListNeuro = list_neuro  # Nombre de neurones cachés 3
        self.model = tf.keras.Sequential()
        self.set_model()

    def forward(self, x):
        return tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        # return probability_model

    # return history
    def train(self, X, y):
        return self.model.fit(X, y, epochs=10, validation_split=0.2)

    # Fonction de prédiction inutile
    def predict(self, xPrediction):
        probability_model = tf.keras.Sequential([self.model,
                                                 tf.keras.layers.Softmax()])
        predictions = probability_model.predict(xPrediction)
        return predictions

    def save_weight(self, name):
        self.model.save_weights(name)

    def load_weight(self, name):
        self.model.load_weights(name)

    # get random without save
    def set_model(self):
        self.model = keras.Sequential(name="my_sequential")
        self.model.add(layers.Dense(self.ListNeuro[0], activation="relu"))
        for i in range(1, len(self.ListNeuro)-1):
            self.model.add(layers.Dense(self.ListNeuro[i], activation="relu"))
        self.model.add(layers.Dense(self.ListNeuro[-1]))

    def compile(self):
        # Compile the model
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="sgd",
            metrics=["accuracy"]
        )

    def evaluate(self, X, y):
        loss, acc = self.model.evaluate(X, y)
        print("Test Loss", loss)
        print("Test Accuracy", acc)
        return loss, acc

"""NN = Neural_Network(2, 6, 3, 1, 2)
error = 1
fail = np.array(["nan"])

while error > 0.0001:
    # for i in range(1000):  # Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    # print("# " + str(i) + "\n")
    # print("Valeurs d'entrées: \n" + str(X))
    # print("Sortie actuelle: \n" + str(y))
    # print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X), 2)))
    # print("\n")
    sortie = NN.train(X, y)
    m = 0
    nb = 0
    # print("error", sortie)
    for x in sortie:
"""
"""
        m += x
        nb += 1

    if nb == 0:
        nb = 1
    error = abs(m / nb)
    print("erreur = ", error)

NN.predict(xPrediction)


finalement tanh ça marche

voir comment enregistrer les poids puis
les ré-ingecter dans un réseau
mettre tanh aulieu de la sigmoid
intégrer à l'ensemble des programe de controle


NN = neural_network(2, 3, 2, 1, 1)

for i in range(1000):  # Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    print("# " + str(i) + "\n")
    print("Valeurs d'entrées: \n" + str(X))
    print("Sortie actuelle: \n" + str(y))
    print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X), 2)))
    print("\n")

    print("sortie", NN.train(X, y))

NN.predict(xPrediction)
"""
"""

class a mettre en haut dans main:

donc 2
- meme que dans bot pour le classement des bots d'entrée (copie colle)
- faire le mixage des meilleurs et organisation ds une liste des bots d'entrée et les bots de sortie


"""



