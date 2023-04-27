import json
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Load folds
folds = []
for i in range(1, 6):
    with open(f'pols_fold_{i}.json') as f:
        fold_data = []
        for line in f:
            obj = json.loads(line)
            fold_data.append(obj)
        folds.append(pd.DataFrame(fold_data))

# Initialize vectorizer and Affinity Propagation objects
vectorizer = CountVectorizer()
ap = AffinityPropagation(damping=0.99, convergence_iter=500)

# Initialize lists to store results
accuracy_list = []
report_list = []

# Perform 5-fold cross-validation
for i in range(5):
    print("Fold ", i)
    # Use ith fold as test set, remaining folds as training set
    test_data = folds[i]
    train_data = pd.concat([folds[j] for j in range(5) if j != i], ignore_index=True)
    
    # Set preference parameter to encourage two clusters
    similarity_matrix = vectorizer.fit_transform(train_data['text']).toarray()
    preference = np.median(similarity_matrix) * 0.001
    ap.set_params(preference=preference)
    
    y_train = ap.fit_predict(similarity_matrix)
    X_test_vect = vectorizer.transform(test_data['text'])
    y_test = ap.predict(X_test_vect.toarray())
    
    accuracy = accuracy_score(y_test, test_data['is_deceptive'])
    report = classification_report(test_data['is_deceptive'], y_test, output_dict=True, zero_division=1)
    
    # Store results for each fold
    print("Accuracy: ", accuracy)
    print(report)
    
    # Plot cluster assignments for the test set
    plt.scatter(X_test_vect.toarray()[:, 0], X_test_vect.toarray()[:, 1], c=y_test)
    plt.title("Test set cluster assignments")
    plt.show()
    
    # Print number of clusters
    n_clusters = len(np.unique(y_train))
    print("Number of clusters:", n_clusters)


# Generate an average report across all folds
#avg_report = {}
#for key in report_list[0].keys():
 #   if isinstance(report_list[0][key], dict):
  #      avg_report[key] = {}
   #     for metric in report_list[0][key].keys():
    #        if key in report_list[0]:
     #           avg_report[key][metric] = np.mean([report[key][metric] for report in report_list])
    #else:
     #   avg_report[key] = np.mean([report[key] for report in report_list])

# Print the average report
#print("Average Classification Report: ")
#print(avg_report)