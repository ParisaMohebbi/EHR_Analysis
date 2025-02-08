"""
Disease Cluster Analysis in Electronic Health Records: Insights into Mortality and Comorbidity Patterns
Submitted to the IISE Annual Conference & Expo 2025
Abstract ID: 6965
Authors: Parisa Vaghfi Mohebbi, Ahmad Salehiyan, Akash Deep
"""

# Import packages
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the CSV data using pandas
file_path = 'Inset Your File path Here'
data = pd.read_csv(file_path)

# Remove rows where drg_codes are NaN
data.dropna(subset=['drg_codes'], inplace=True)

# Evaluate the string representation of lists
data['drg_codes'] = data['drg_codes'].apply(lambda x: eval(x))

# Remove duplicates from each list of drg_codes
data['drg_codes'] = data['drg_codes'].apply(lambda x: list(set(x)))

# Calculate the number of items in each transaction
data['num_items'] = data['drg_codes'].apply(len)

# Compute the average number of items per transaction
average_items_per_transaction = data['num_items'].mean()

print("Average Items per Transaction:", average_items_per_transaction)

# Count occurrences of each number of items
item_counts = data['num_items'].value_counts().sort_index()

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(item_counts.index, item_counts.values, color='skyblue')
plt.xlabel('Number of Items per Transaction')
plt.ylabel('Number of Occurrences')
plt.title('Distribution of Number of Items per Transaction')
plt.xticks(item_counts.index)  # Set x-ticks to show all item counts
plt.show()

# Convert to a DataFrame for a table format
table = pd.DataFrame({'Number of Items per Transaction': item_counts.index,
                      'Number of Occurrences': item_counts.values})

print(table)

# Export the DataFrame of rules to an Excel file
table.to_excel('NumberـItemsـperـTransaction.xlsx', index=False)

# Flatten the 'drg_codes' lists to get all items across all transactions
all_items = [item for sublist in data['drg_codes'] for item in sublist]

# Count occurrences of each unique item
item_counts = pd.Series(all_items).value_counts().reset_index()
item_counts.columns = ['Item', 'Frequency']

# Display the table
item_counts

# Prepare the data for FP-Growth by converting each transaction into a list of items
transactions = data['drg_codes'].tolist()

# Use TransactionEncoder to transform the data into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

data

# Run FP-Growth to find frequent itemsets
min_support = 0.001  # Set a minimum support threshold as needed
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

# Generate association rules based on the frequent itemsets
min_confidence = 0.1
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence, support_only=False, num_itemsets= 223452)

# Display the frequent itemsets and rules
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)

# Export the DataFrame of rules to an Excel file
rules.to_excel('association_rules.xlsx', index=False)

# Calculate the number of items in each transaction
data['num_items'] = data['drg_codes'].apply(len)

# Filter transactions to include only those with 5 or more items
filtered_data = data[data['num_items'] >= 5]

filtered_data

# Use TransactionEncoder to transform the data into a one-hot encoded DataFrame
transactions = filtered_data['drg_codes'].tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Run FP-Growth to find frequent itemsets
min_support = 0.01  # Set a minimum support threshold as needed
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

# Generate association rules based on the frequent itemsets
min_confidence = 0.1
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence, num_itemsets = 179524 )

# Display the frequent itemsets and rules
print("Filtered Frequent Itemsets:\n", frequent_itemsets)
print("\nFiltered Association Rules:\n", rules)

# Export the DataFrame of rules to an Excel file
rules.to_excel('association_rules_5.xlsx', index=False)





