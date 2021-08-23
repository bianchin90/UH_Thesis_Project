import pandas as pd

population = pd.read_excel("Georeferencing/popolazione_comuni.xlsx")
cities = pd.read_excel("Georeferencing/Elenco-comuni-italiani.xls")

print(population.head())