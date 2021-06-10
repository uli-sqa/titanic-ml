from TitanicML import *


# lade die Titanic-Passagierliste
titanic_dataset = pd.read_csv("titanic.csv", index_col=0)
print(titanic_dataset.columns)

# zeige die stärksten Korrelationen
correlations = titanic_dataset.corr()
print(correlations)

# erstelle Histogramme über die Werte der einzelnen Spalten
titanic_dataset.hist()
plt.tight_layout()

# erstelle ausgewählte Diagramme
TitanicML.plot_bar(titanic_dataset, "Pclass", "Survived")
TitanicML.plot_bar_hue(titanic_dataset, "Pclass", "Survived", "Sex")

plt.show()
