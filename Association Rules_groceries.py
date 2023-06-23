


# Install the required packages if not available
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from sqlalchemy import create_engine
import pickle

list1 = ['kumar', '360DigiTMG', 2019, 2022]

print(list1)

print(len(list1))

print(list1[0])
print(list1[3])
print(list1[:3])

# Reversing list
print(list1[::-1])

del(list1[0])

print(list1)

# Create a tuple dataset
tup1 = ('kumar', '360DigiTMG', 2019, 2022)

tup2 = (1, 2, 3, 4, 5, 6, 7)

print(tup1)

print(tup2)

print(tup1[0])

print(tup2[1:5])


tup1[1]   

del(tup1[1]) # We cannot delete individual items of tuple as it is immutable.


# Connecting to sql by creating sqlachemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = "WorkBench1", db = "retail_db")) # database

user = 'root'
pw = 'WorkBench1'
db = 'retail_db'


# Read csv file 
data = pd.read_csv(r"C:\Users\Karuna Singh\Downloads\Mar.23\Datasets\clustering\Association Rules\groceries.csv" , sep = ';', header = None)
data.head()

# data_r = pd.read_csv(r"C:\Users\Karuna Singh\Downloads\Mar.23\Datasets\clustering\Association Rules\groceries.csv" , header = None)

data.to_sql('groceries', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# read data from database
sql = 'select * from groceries;'
groceries = pd.read_sql_query(sql, con = engine)

groceries.head()

# Suppress the Warnings
import warnings
warnings.filterwarnings("ignore")

# Convert the groceries into list
groceries = groceries.iloc[:, 0].to_list()
groceries

# Extract items from the transactions
groceries_list = []

for i in groceries:
   groceries_list.append(i.split(","))

print(groceries_list)


# Removing null values from list
groceries_list_new = []

for i in  groceries_list:
   groceries_list_new.append(list(filter(None, i)))

print(groceries_list_new)


# TransactionEncoder: Encoder for transaction data in Python lists
# Encodes transaction data in the form of a Python list of lists,
# into a NumPy array

TE = TransactionEncoder()
X_1hot_fit = TE.fit(groceries_list)


# import pickle
pickle.dump(X_1hot_fit, open('TE.pkl', 'wb'))

import os
os.getcwd()

X_1hot_fit1 = pickle.load(open('TE.pkl', 'rb'))

X_1hot = X_1hot_fit1.transform(groceries_list) 

print(X_1hot)


transf_df = pd.DataFrame(X_1hot, columns = X_1hot_fit1.columns_)
transf_df
transf_df.shape


### Elementary Analysis ###
# Most popular items
count = transf_df.loc[:, :].sum()

# Generates a series
pop_item = count.sort_values(0, ascending = False).head(10)

# Convert the series into a dataframe 
pop_item = pop_item.to_frame() # type casting

# Reset Index
pop_item = pop_item.reset_index()
pop_item

pop_item = pop_item.rename(columns = {"index": "items", 0: "count"})
pop_item

# Data Visualization
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 6) # rc stands for runtime configuration 
plt.style.use('dark_background')
pop_item.plot.barh()
plt.title('Most popular items')
plt.gca().invert_yaxis() # gca means "get current axes"

help(apriori)

# Itemsets
frequent_itemsets = apriori(transf_df, min_support = 0.0075, max_len = 4, use_colnames = True)
frequent_itemsets


# Most frequent itemsets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets

# Association Rules
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(10)

rules.sort_values('lift', ascending = False).head(10)


# Handling Profusion of Rules (Duplication elimination)
def to_list(i):
    return (sorted(list(i)))

# Sort the items in Antecedents and Consequents based on Alphabetical order
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

# Sort the merged list of items - transactions
ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

# No duplication of transactions
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

# Capture the index of unique item sets
index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    
index_rules

# Rules without any redudancy 
rules_no_redundancy = rules.iloc[index_rules, :]
rules_no_redundancy


# Sorted list and top 10 rules 
rules10 = rules_no_redundancy.sort_values('lift', ascending = False).head(10)
rules10

rules10.plot(x = "support", y = "confidence", c = rules10.lift, 
             kind = "scatter", s = 12, cmap = plt.cm.coolwarm)


# Store the rules on to SQL database
# Database do not accepting frozensets

# Removing frozenset from dataframe
rules10['antecedents'] = rules10['antecedents'].astype('string')
rules10['consequents'] = rules10['consequents'].astype('string')

rules10['antecedents'] = rules10['antecedents'].str.removeprefix("frozenset({")
rules10['antecedents'] = rules10['antecedents'].str.removesuffix("})")

rules10['consequents'] = rules10['consequents'].str.removeprefix("frozenset({")
rules10['consequents'] = rules10['consequents'].str.removesuffix("})")

rules10.to_sql('groceries_ar', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


