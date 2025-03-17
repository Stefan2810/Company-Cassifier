# Company-Classifier

## Code structure:

### 1.Load and Preprocess Data:
– Read the company list and taxonomy CSV files.
– Combine key text fields for each company (description, business tags, sector, category, niche) into one string and convert all text to lowercase.
– Preprocess the taxonomy labels in the same way.

### 2.Vectorization and Similarity Calculation:
– Fit a TF-IDF vectorizer on the company text (to capture the vocabulary used by companies) and then transform the taxonomy labels using this same vectorizer.
– Compute the cosine similarity between each company and each taxonomy label.

### 3.Label Assignment:
– For each company, if one or more taxonomy labels have a similarity score above a set threshold (here, 0.1), assign those labels.
– If no taxonomy label exceeds the threshold, assign the best-matching label (the one with the highest similarity).

### 4.Output:
– The final output is a CSV file containing only the new column “insurance_label” that holds the assigned taxonomy labels for each company.

## Working approach and overview:

### Thinking process
- Data processing: columns concatenation, lowercase filtering and filling null values with "" strings was made to have data consistency
- Vectorization is made to convert the information into numerical vectors, making it easier for the algorithm to handle the data
- The Similarity Measurement is meant to see how closely each company’s description matches each taxonomy label
- The Label Assigment makes its decisions based on a given treshold

### Strengths and Weaknesses
- The algorithm is simple, easy to read and understand. The TF-IDF and cosine similarity are methods that can be used for larges datasets.
- As Weaknesses, the program does not capture deeper semantic relationships, being only based on word frequency. Another aspect is that the predictions are made on the assumption that all pieces contribute equally to the companies profile, which might not always be true.

### Conclusions
- The current solution is usable in a moderate dataset context
- For the future, the feature weighting might be done a little different
