import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

dfShared = pd.read_csv('shared_articles.csv')
dfUser = pd.read_csv('users_interactions.csv')

dfShared = dfShared[dfShared['eventType'] == 'CONTENT SHARED']
dfShared = dfShared[['contentId', 'title', 'text']]
dfUser = dfUser[dfUser['eventType'] == 'VIEW']
dfUser = dfUser[['personId', 'contentId']]

numInteraction = dfUser['personId'].value_counts()
sufUsers = numInteraction[numInteraction >= 3].index
dfUser = dfUser[dfUser['personId'].isin(sufUsers)]
df_merged = pd.merge(dfUser, dfShared, on='contentId', how='inner')

train_data, test_data = train_test_split(df_merged, test_size=0.2, random_state=42, stratify=df_merged['personId'])

userIndex = pd.Index(train_data['personId'].unique(), name='personId')
itemIndex = pd.Index(train_data['contentId'].unique(), name='contentId')

userMap = {user_id: index for index, user_id in enumerate(userIndex)}
itemMap = {item_id: index for index, item_id in enumerate(itemIndex)}

train_data['user_index'] = train_data['personId'].map(userMap)
train_data['item_index'] = train_data['contentId'].map(itemMap)

user_item_matrix = csr_matrix((np.ones(len(train_data)), (train_data['user_index'], train_data['item_index'])),
                              shape=(len(userIndex), len(itemIndex)))

cosMat = cosine_similarity(user_item_matrix)

user_id = 344280948527967603

uArticle = train_data[train_data['personId'] == user_id]['contentId']
uArticleindex = [itemMap[item_id] for item_id in uArticle]

user_profile = user_item_matrix[userMap[user_id], uArticleindex].mean(axis=0).reshape(1, -1)

user_row = user_item_matrix[userMap[user_id], :]
similarUsers = cosine_similarity(user_row, user_item_matrix).ravel()

mostSimilar = np.argsort(similarUsers)[::-1]
recommended_articles = [itemIndex[i] for i in mostSimilar if i not in uArticleindex][:5]

print(f"User Profile: {user_id}")

print("Recommended Articles:")
for t in recommended_articles:
    z = dfShared[dfShared['contentId'] == t]
    print(f"Title: {z['title'].values[0]}")
