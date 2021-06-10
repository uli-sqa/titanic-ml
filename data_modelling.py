import pandas as pd
import matplotlib.pyplot as plt
from TitanicML import TitanicML


# Daten laden
titanic_dataset = pd.read_csv("titanic.csv", index_col=0)

# Daten aufbereiten
data, result = TitanicML.prepare_data(titanic_dataset)
print(data.columns)
print(data.head(5))

# Aufteilung in Training-Set und Test-Set (80:20)
data_train, data_test, result_train, result_test = TitanicML.split_data(data, result, split=0.2)

# Das Modell trainieren und prüfen
model = TitanicML.train_model(data_train, result_train)
TitanicML.evaluate_model(model, data_test, result_test)

# Das Modell erklärt sich
TitanicML.explain_model(model, data_train)

# Wie war der Trainings-Verlauf
TitanicML.visualize_training_progress(model, data_train, result_train, data_test, result_test)
plt.show()




