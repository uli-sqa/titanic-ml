import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from eli5 import explain_weights_df
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class TitanicML:

    @staticmethod
    def plot_bar(data, x, y):
        """
        Zeichnet ein Balkendiagramm (2 Balken)
        :param x: Spaltename für X-Achse
        :param y: Spaltenname für Y-Achse
        """

        plt.figure()
        f = sns.barplot(x=x, y=y, data=data, ci=0)
        f.set(title=x + " rate by " + y)

    @staticmethod
    def plot_bar_hue(data, x, y, z):
        """
        Zeichnet ein Balkendiagramm mit unterteilten Balken
        :param x: Spaltenname für X-Achse
        :param y: Spaltenname für Y-Achse
        :param z: Spaltename für die Unterteilung der Balken
        """
        plt.figure()
        f = sns.barplot(x=x, y=y, hue=z, data=data, ci=0)
        f.set(title=x + " rate by " + y)

    @staticmethod
    def prepare_data(data, includeColumns=()):
        """
        Daten für ML aufbereiten
        * Sex => Male = 0, Female = 1
        * Familiengröße = SibSp + Parch + 1
        * PClass => Class_1, Class_2, Class_3 jeweils mit 0 oder 1 befüllt
        :return: datenset für ML und resultset zur Prüfung
        """
        sex_encoder = LabelEncoder()
        data["Gender"] = sex_encoder.fit_transform(data["Sex"])
        data["Family_Size"] = data.SibSp + data.Parch + 1
        classes = pd.get_dummies(data.Pclass, prefix="Class")
        ports = pd.get_dummies(data.Embarked, prefix="Port")

        # Datenset vorbereiten: Survived darf nicht im ML-Datensatz enthalten sein
        columns = [
            data.Gender,
            data.Family_Size,
            classes,
            ports]
        for column in includeColumns:
            columns.append(data[column])

        preprocessed_data = pd.concat(columns, axis=1)
        result_data = data["Survived"]
        return preprocessed_data, result_data

    @staticmethod
    def split_data(x, y, split=0.5):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=42)
        pctg_size = round(x_train.shape[0] / x.shape[0] * 100)
        print(f"Training set is {pctg_size}% of the original dataset")
        return x_train, x_test, y_train, y_test

    @staticmethod
    def train_model(x_train, y_train):
        clf = DecisionTreeClassifier(max_depth=6, random_state=42)
        clf.fit(x_train, y_train)
        print("The model has been trained")
        return clf

    @staticmethod
    def evaluate_model(model, x_test, y_test):
        preds = model.predict(x_test)
        print("Evaluation model")
        score = accuracy_score(y_test, preds) * 100
        print(f"The model achieved {round(score, 2)}% accuracy on the test dataset")

    @staticmethod
    def visualize_training_progress(model, x_train, y_train, x_test, y_test):
        sizes = [2, 8, 10, 12, 16, 20, 24, 32, 40, 52, 64, 128, 256, 512, 720]
        train_scores = []
        test_scores = []

        for size in sizes:
            x_tr = x_train[:size]
            y_tr = y_train[:size]

            model.fit(x_tr, y_tr)
            train_scores.append(accuracy_score(y_tr, model.predict(x_tr)))
            test_scores.append(accuracy_score(y_test, model.predict(x_test)))

        fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=100)
        plt.grid(False)

        ax.plot(sizes, test_scores, color="black", label="Test Score", lw=5)[0]
        ax.plot(sizes, train_scores, color="grey", label="Train Score", lw=3)
        plt.xlabel("Dataset Size")
        plt.ylabel("Accuracy")
        plt.legend()

    @staticmethod
    def explain_model(model, X_train):
        weights = explain_weights_df(model, feature_names=X_train.columns.tolist())
        print(weights.head(10))
        f = sns.barplot(x="feature", y="weight", data=weights, ci=0)
        f.set(title="Model Explanation")