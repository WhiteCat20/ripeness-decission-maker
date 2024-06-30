import numpy as np

def calculate_weights(accuracies):
    return accuracies / np.sum(accuracies)

def weighted_majority_voting(weights, predictions):
    weighted_votes = np.dot(weights, predictions)
    return np.argmax(weighted_votes)

# Example accuracies for three classifiers
accuracies = np.array([1, 0.4])

# Calculate weights
weights = calculate_weights(accuracies)

# Example predictions from the classifiers for a test instance
# Let's assume we have three classes: 0, 1, 2
predictions = np.array([
    [1],  # Classifier 1 predicts class 0
    [0]  # Classifier 2 predicts class 1
])

# Perform weighted majority voting
predicted_class = weighted_majority_voting(weights, predictions)

print("Classifier Weights: ", weights)
print("Weighted Votes: ", np.dot(weights, predictions))
print("Final Predicted Class: ", predicted_class)
