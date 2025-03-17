import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

companies = pd.read_csv("ml_insurance_challenge.csv")
taxonomy = pd.read_csv("insurance_taxonomy.csv")

# -combine text fields and convert to lowercase:
# -.fillna("") --> fills empty spaces
# -the lambda function converts the text fields to lower case
companies['combined_text'] = companies[['description', 'business_tags', 'sector', 'category', 'niche']] \
    .fillna("") \
    .apply(lambda x: " ".join(x.astype(str)).lower(), axis=1)
taxonomy['label'] = taxonomy['label'].str.lower()

# Vectorize: Fit on companies text and transform taxonomy labels
vec = TfidfVectorizer()
company_vec = vec.fit_transform(companies['combined_text'])
taxonomy_vec = vec.transform(taxonomy['label'])

# Compute cosine similarity and set a threshold
sim_matrix = cosine_similarity(company_vec, taxonomy_vec)
threshold = 0.1

# Function: Assign labels if above threshold; otherwise, fallback to best match
def assign_labels(sim_row):
    indices = np.where(sim_row > threshold)[0]
    if not len(indices):
        indices = [np.argmax(sim_row)]
    return taxonomy['label'].iloc[indices].tolist()

# Assign labels for each company
companies['insurance_label'] = [assign_labels(row) for row in sim_matrix]

# Output only the labels
companies[['insurance_label']].to_csv("company_labels.csv", index=False)

# I chose to create a different csv file so that the labels could be easier to see
print("Classification complete. Labels saved to company_labels.csv")
