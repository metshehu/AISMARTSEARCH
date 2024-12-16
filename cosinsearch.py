import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Generate random data
num_samples = 100
dimensionality = 3
data = np.random.rand(num_samples, dimensionality)
print(data)
for i in data


# Define a query vector
query_vector = np.random.rand(dimensionality)

# Calculate cosine similarities between the query vector and the dataset
similarities = cosine_similarity(data, [query_vector])

# Find the most similar vector
most_similar_index = np.argmax(similarities)
most_similar_vector = data[most_similar_index]

print(f"Most similar vector: {most_similar_vector}")
