#"C:\Users\deela\Downloads\noc_regions.csv"
#"C:\Users\deela\Downloads\athlete_events.csv"

# Numpy
import numpy as np

# Pandas
import pandas as pd

# Matplotlib
import matplotlib.pyplot as plt

# Seaborn
import seaborn as sns

# RegEx
import re

athlete = pd.read_csv("C:\\Users\\deela\\Downloads\\athlete_events.csv")
noc = pd.read_csv("C:\\Users\\deela\\Downloads\\noc_regions.csv")
print(athlete)
print(noc)

ath = athlete.copy()
nat = noc.copy()

print(ath.describe())
print("\n null values sum \n  ",ath.isnull().sum())

region_col = ath['NOC'].map(nat.set_index('NOC')['region'])
print(region_col)
ath.insert(7, 'region', region_col)

print(ath['region'])

ath.drop("NOC", inplace = True, axis =1)

ath['Age'].fillna(ath.Age.mean(), inplace=True)
ath['Height'].fillna(ath.Height.mean(), inplace=True)
ath['Weight'].fillna(ath.Weight.mean(), inplace=True)

print(ath.iloc[[147]])

print(ath.isnull().sum())

print(ath['Medal'].unique())
# Replace 
# NaN with 0
# Gold with 1
# Silver with 2
# Bronze with 3

ath['Medal'].replace(np.nan,0, inplace=True)


ath['Medal'].replace('Gold',1, inplace=True)
ath['Medal'].replace("Silver",2, inplace=True)
ath['Medal'].replace("Bronze",3, inplace=True)
print(ath['Medal'])

print(ath['Medal'].value_counts())
ath.Medal = ath.Medal.astype(int)

# Columns to be drop off -

# 1. region
# 2. Games

ath.drop(["region","Games"],axis = 1,inplace =True)

for i, j in zip(ath['Sport'], range(len(ath['Event']))):
    ath.at[j, 'Event'] = re.sub(f"{i}\s", "", ath.at[j, 'Event']) #\s is a regular expression pattern that matches any whitespace character in a string.

print(ath)



ath.Age = ath.Age.astype(int)

# Export to JSON
#ath.to_json("athletes_dataset.json")

# Export to Excel
#ath.to_excel("athletes_dataset.xlsx")

# Export to .CSV
#ath.to_csv("athletes_dataset.csv")

x = ath.Height
y = ath.Weight
plt.scatter(x,y)
plt.title("Height V/s Weight")
plt.xlabel("Height")
plt.ylabel("Weight")
#plt.show()

print((ath.groupby(ath['Team'])['Medal'].sum()).sort_values(ascending=False).head(5))


#print(ath.groupby(ath['Team'])['Medal'].sum())