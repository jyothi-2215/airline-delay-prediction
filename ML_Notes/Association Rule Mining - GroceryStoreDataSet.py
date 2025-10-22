from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('GroceryStoreDataSet.csv', names = ['products'], sep = ',')
df.head()


data = list(df["products"].apply(lambda x:x.split(",") ))

# Apriori Algorithm and One-Hot Encoding
from mlxtend.preprocessing import TransactionEncoder
# df1 = pd.get_dummies(df, dtype = float)
#
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df = pd.DataFrame(a_data,columns=a.columns_)
change = {False:0, True:1}
df = df.replace(change)
print(df)
# ===============================
# Applying Apriori and Resulting
# ==============================
#set a threshold value for the support value and calculate the support value.
df = apriori(df, min_support = 0.2, use_colnames = True, verbose = 1)
print(df)
# I chose the 60% minimum confidence value. In other words, when product X
# is purchased, we can say that the purchase of product Y is 60% or more.
#Let's view our interpretation values using the Associan rule function.
df_ar = association_rules(df, metric = "confidence", min_threshold = 0.6)
df_ar = df_ar.sort_values(['confidence','lift'], ascending = [False, False])
print(df_ar.to_string())
