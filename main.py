import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def get_article_recommendations(article_id, cosine_similarities, data):

    article_index = data[data['contentId'] == article_id].index[0]

    similarity_scores = list(enumerate(cosine_similarities[article_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_scores = similarity_scores[1:6]

    article_indices = [score[0] for score in top_scores]

    return data['contentId'].iloc[article_indices]


if __name__ == '__main__':

    data = pd.read_csv('shared_articles.csv')

    data = data[data['eventType'] == 'CONTENT SHARED']
    data['text'] = data['title'] + ' ' + data['text']
    data = data.drop_duplicates(subset='contentId', keep='first')
    data = data.dropna(subset=['text'])

    vectorizer = TfidfVectorizer(stop_words='english')

    tfidf = vectorizer.fit_transform(data['text'])

    cos_sim = linear_kernel(tfidf, tfidf)

    # example input
    article_id = 2448026894306402386

    recommendations = get_article_recommendations(article_id, cos_sim, data)

    print(f"Recommendations for article {article_id}:")

    for article in recommendations:
        print(f"Article ID: {article}")
