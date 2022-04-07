import numpy as np

# merci à ceux qui ont expliquer le fonctionnement de ceux code


# x_entrer = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [4, 1.5]),
#                    dtype=float)  # données d'entrer
# y = np.array(([1], [0], [1], [0], [1], [0], [1], [0]), dtype=float)  # données de sortie /  1 = rouge /  0 = bleu

# Changement de l'échelle de nos valeurs pour être entre 0 et 1
# x_entrer = x_entrer / np.amax(x_entrer, axis=0)  # On divise chaque entré par la valeur max des entrées

# On récupère ce qu'il nous intéresse
# X = np.split(x_entrer, [8])[0]  # Données sur lesquelles on va s'entrainer, les 8 premières de notre matrice
# xPrediction = np.split(x_entrer, [8])[1]  # Valeur que l'on veut trouver


# Notre classe de réseau neuronal
class neural_network(object):
    def __init__(self, nb_neuro_entre, nb_neuro_hidden, nb_neuro_hidden2, nb_neuro_output, nb_hidden):

        # Nos paramètres
        self.inputSize = nb_neuro_entre  # Nombre de neurones d'entrer 2
        self.hiddenSize = nb_neuro_hidden  # Nombre de neurones cachés 3
        self.hiddenSize2 = nb_neuro_hidden2  # Nombre de neurones cachés2 3
        self.outputSize = nb_neuro_output  # Nombre de neurones de sortie 1
        self.nb_hidden = nb_hidden

        # if self.nb_hidden == 2:
        # Nos poids
        self.W1 = np.random.randn(self.inputSize,
                                  self.hiddenSize)  # (2x3) Matrice de poids entre les neurones d'entrer et cachés

        self.W2 = np.random.randn(self.hiddenSize,
                                  self.hiddenSize2)  # (3x2) Matrice de poids entre les neurones cachés et sortie

        self.W3 = np.random.randn(self.hiddenSize2,
                                  self.outputSize)  # (2x1) Matrice de poids entre les neurones cachés et sortie
        """else:
            # Nos poids
            self.W1 = np.random.randn(self.inputSize,
                                      self.hiddenSize)  # (2x3) Matrice de poids entre les neurones d'entrer et cachés
            self.W2 = np.random.randn(self.hiddenSize,
                                      self.outputSize)  # (3x1) Matrice de poids entre les neurones cachés et sortie
        """
        """print("W1", self.W1.shape)
        print("W1", self.W1)
        print("w2", self.W2.shape)
        print("w2", self.W2)
        print("w3", self.W3.shape)
        print("w3", self.W3)"""

    # Fonction de propagation avant
    def forward(self, x):
        # print("input", self.inputSize)
        # print("x", len(x))
        # print("hidden size w1", self.W1.shape)
        # if self.nb_hidden == 2:
        self.z = np.dot(x, self.W1)  # Multiplication matricielle entre les valeurs d'entrer et les poids W1
        self.z2 = self.sigmoid(self.z)  # Application de la fonction d'activation (Sigmoid)
        self.z3 = np.dot(self.z2, self.W2)  # Multiplication matricielle entre les valeurs d'entrer et les poids W2
        self.z4 = self.sigmoid(self.z3)  # Application de la fonction d'activation (Sigmoid)
        self.z5 = np.dot(self.z4, self.W3)  # Multiplication matricielle entre les valeurs cachés et les poids W2
        o = self.sigmoid(
            self.z5)  # Application de la fonction d'activation, et obtention de notre valeur de sortie final

        return o
        """else:
            self.z = np.dot(x, self.W1)  # Multiplication matricielle entre les valeurs d'entrer et les poids W1
            self.z2 = self.sigmoid(self.z)  # Application de la fonction d'activation (Sigmoid)
            self.z3 = np.dot(self.z2, self.W2)  # Multiplication matricielle entre les valeurs cachés et les poids W2
            o = self.sigmoid(
                self.z3)  # Application de la fonction d'activation, et obtention de notre valeur de sortie final
            return o
            # entre 0 et 1"""

    # Fonction d'activation
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    # Dérivée de la fonction d'activation
    def sigmoidPrime(self, s):
        return s * (1 - s)

    """# fonction d'activation
    def sigmoid(self, s):
        t = np.tanh(s)
        return t

    # dérivée de la fonction d'activation TANH
    def sigmoidPrime(self, s):
        dt = 1 - np.tanh(s) ** 2
        return dt"""

    # Fonction de rétropropagation
    def backward(self, X, y, o):

        # if self.nb_hidden == 2:
        self.o_error = y - o  # Calcul de l'erreur
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # Application de la dérivée de la sigmoid à cette erreur

        self.z4_error = self.o_delta.dot(self.W3.T)  # Calcul de l'erreur de nos neurones cachés
        self.z4_delta = self.z4_error * self.sigmoidPrime(self.z4)
        # Application de la dérivée de la sigmoid à cette erreur

        self.z2_error = self.z4_delta.dot(self.W2.T)  # Calcul de l'erreur de nos neurones cachés
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        # Application de la dérivée de la sigmoid à cette erreur

        self.W1 += X.T.dot(self.z2_delta)  # On ajuste nos poids W1
        self.W2 += self.z2.T.dot(self.z4_delta)  # On ajuste nos poids W2
        self.W3 += self.z4.T.dot(self.o_delta)  # On ajuste nos poids W3
        """else:
            self.o_error = y - o  # Calcul de l'erreur
            self.o_delta = self.o_error * self.sigmoidPrime(o)  # Application de la dérivée de la sigmoid à cette erreur

            self.z2_error = self.o_delta.dot(self.W2.T)  # Calcul de l'erreur de nos neurones cachés
            self.z2_delta = self.z2_error * self.sigmoidPrime(
                self.z2)  # Application de la dérivée de la sigmoid à cette erreur

            self.W1 += X.T.dot(self.z2_delta)  # On ajuste nos poids W1
            self.W2 += self.z2.T.dot(self.o_delta)  # On ajuste nos poids W2"""
        return self.o_error

    # Fonction d'entrainement, inutile
    def train(self, X, y):

        o = self.forward(X)
        self.backward(X, y, o)
        return o

    # Fonction de prédiction inutile
    def predict(self, xPrediction):

        # print("Donnée prédite apres entrainement: ")
        # print("Entrée : \n" + str(xPrediction))
        # print("Sortie : \n" + str(self.forward(xPrediction)))
        if self.forward(xPrediction) < 0.5:  # 0.5 pr sigmoid
            print("La fleur est BLEU ! \n")
        else:
            print("La fleur est ROUGE ! \n")

    def save_w(self):
        pass

    def get_weight(self):
        if self.nb_hidden == 2:
            self.save_w()
            return [self.W1, self.W2, self.W3]
        else:
            self.save_w()
            return [self.W1, self.W2]

    def set_weight(self, w1=None, w2=None, w3=None):
        if w1 is not None:
            self.W1 = np.array(w1)

        if w2 is not None:
            self.W2 = np.array(w2)

        if w3 is not None:
            self.W3 = np.array(w3)

    # get random without save
    def set_random(self):

        # if self.nb_hidden == 2:
        # Nos poids
        W1 = np.random.randn(self.inputSize,
                             self.hiddenSize)  # (2x3) Matrice de poids entre les neurones d'entrer et cachés
        W2 = np.random.randn(self.hiddenSize,
                             self.hiddenSize2)  # (3x2) Matrice de poids entre les neurones cachés et sortie
        W3 = np.random.randn(self.hiddenSize2,
                             self.outputSize)  # (2x1) Matrice de poids entre les neurones cachés et sortie
        return [W1, W2, W3]
        """elif self.nb_hidden == 1:
            # Nos poids
            W1 = np.random.randn(self.inputSize,
                                 self.hiddenSize)  # (2x3) Matrice de poids entre les neurones d'entrer et cachés
            W2 = np.random.randn(self.hiddenSize,
                                 self.outputSize)  # (3x1) Matrice de poids entre les neurones cachés et sortie
            return [W1, W2]"""


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
"""if not x < 0 and y[nb] < 0 or x > 0 and y[nb] > 0:
            m += 1  # y[nb] - x
            nb += 1
        elif x == 0:
            m += 9999
            nb += 1
        print("x", x, "y", y[nb])"""
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
