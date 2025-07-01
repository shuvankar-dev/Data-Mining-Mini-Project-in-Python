#!/usr/bin/env python
# coding: utf-8

# ## Load Required Libraries

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df = pd.ExcelFile("data mining supporting file.xlsx")


# In[11]:


sheet_2018 = df.parse('Lobster Data 2018')
sheet_2019 = df.parse('Lobster Data 2019')


# In[15]:


# Add 'Year' column
sheet_2018['Year'] = 2018
sheet_2019['Year'] = 2019


# In[17]:


# Merge datasets
merged_df = pd.concat([sheet_2018, sheet_2019], ignore_index=True)


# In[19]:


merged_df.head()


# In[21]:


merged_df.tail()


# In[23]:


merged_df.info()


# In[25]:


merged_df.isnull().sum()


# In[29]:


merged_df["Length(mm)"] = merged_df["Length(mm)"].fillna(0)
merged_df["Diameter(mm)"] = merged_df["Diameter(mm)"].fillna(0)
merged_df["Height(mm)"] = merged_df["Height(mm)"].fillna(0)
merged_df["WholeWeight(g)"] = merged_df["WholeWeight(g)"].fillna(0)
merged_df["ShuckedWeight(g)"] = merged_df["ShuckedWeight(g)"].fillna(0)
merged_df["SellWeight(g)"] = merged_df["SellWeight(g)"].fillna(0)
merged_df["Spots"] = merged_df["Spots"].fillna(0)


# ### Clean Data

# In[31]:


merged_df.isnull().sum()


# In[36]:


merged_df.describe()


# ### Filler dataset ('Sex' Column)

# In[60]:


merged_df = merged_df[merged_df['Sex'].isin(['M', 'F', 'I'])] # Remove '0'


# In[62]:


print("Unique values in Sex column:", merged_df['Sex'].unique())


# ## Visualization

# In[64]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[66]:


# seaborn theme
sns.set(style="whitegrid")


# In[68]:


# Count of Lobsters by Sex per Year
plt.figure(figsize=(8, 5))
sns.countplot(data=merged_df, x='Sex', hue='Year', palette='Set2')
plt.title("Count of Lobsters by Sex and Year")
plt.tight_layout()
plt.show()


# In[70]:


# Boxplot - Whole Weight by Sex
plt.figure(figsize=(8, 5))
sns.boxplot(data=merged_df, x='Sex', y='WholeWeight(g)', palette='Set3')
plt.title("Distribution of Whole Weight by Sex")
plt.tight_layout()
plt.show()


# In[72]:


# Average Whole Weight by Year and Sex (Time-Aware)
avg_weight = merged_df.groupby(['Year', 'Sex'])['WholeWeight(g)'].mean().unstack()
avg_weight.plot(kind='bar', figsize=(8, 5), title='Average Whole Weight by Year & Sex')
plt.ylabel("Average Whole Weight (g)")
plt.tight_layout()
plt.show()


# In[74]:


# Length vs Weight colored by Sex(Scatterplot)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=merged_df, x='Length(mm)', y='WholeWeight(g)', hue='Sex', palette='cool')
plt.title("Length vs Whole Weight by Sex")
plt.tight_layout()
plt.show()


# In[76]:


# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(merged_df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix of Lobster Variables")
plt.tight_layout()
plt.show()


# ### Data Preprocessing

# In[80]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[82]:


model_df = merged_df[merged_df['Sex'].isin(['M', 'F', 'I'])].copy()


# In[86]:


# Data Encodeing
label_encoder = LabelEncoder()
model_df['Sex_encoded'] = label_encoder.fit_transform(model_df['Sex'])


# In[88]:


# Define features and target
X = model_df.drop(columns=['Sex', 'Sex_encoded', 'Year'])  # Keep clean features
y = model_df['Sex_encoded']


# In[90]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ## Random Forest

# In[97]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[99]:


# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# In[101]:


y_pred = rf_model.predict(X_test)


# In[103]:


accuracy = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Accuracy: {accuracy:.2f}")


# In[107]:


# 4. Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# In[111]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ## KMEANS CLUSTERING (Unsupervised)

# In[114]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[116]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[118]:


# Run KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)


# In[120]:


model_df['Cluster'] = clusters


# In[122]:


# Plot clusters
plt.figure(figsize=(7, 5))
sns.scatterplot(data=model_df, x='Length(mm)', y='WholeWeight(g)', hue='Cluster', palette='Set2')
plt.title("KMeans Clustering of Lobsters")
plt.tight_layout()
plt.show()

