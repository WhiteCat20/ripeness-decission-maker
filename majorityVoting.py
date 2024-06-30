import numpy as np
from model_citra import model_1, model_2, model_3
from model_aroma import model_4, model_5, model_6


def calculate_weights(accuracies):
    total_accuracy = sum(accuracies)
    weights = [accuracy / total_accuracy for accuracy in accuracies]
    return weights


def weighted_majority_vote(predictions, accuracies):
    # Calculate weights based on accuracies
    weights = calculate_weights(accuracies)

    combined_predictions = []

    # Perform weighted majority voting
    for preds in zip(*predictions):
        combined_prediction = sum(
            weight * pred for weight, pred in zip(weights, preds))
        combined_predictions.append(1 if combined_prediction >= 0.5 else 0)
        # combined_predictions.append(combined_prediction)

    return combined_predictions


# Example usage:

nomor_matang = 4
nomor_mentah = 11
# file = f'ripe/{nomor_matang} matang'
file = f'unripe/{nomor_mentah} mentah'

model1 = model_1(f'data gambar/{file}.png')
model2 = model_2(f'data gambar/{file}.png')
model3 = model_3(f'data gambar/{file}.png')
model4 = model_4(f'data aroma/{file}.csv')
model5 = model_5(f'data aroma/{file}.csv')
model6 = model_6(f'data aroma/{file}.csv')


predictions = [
    [model1],
    [model2],
    [model3],
    [model4],
    [model5],
    [model6]
]

accuracies = [
    0.73,  # Example accuracy of model 1
    0.75,  # Example accuracy of model 2
    0.73,  # Example accuracy of model 3
    1.0,  # Example accuracy of model 4
    1.0,  # Example accuracy of model 5
    0.94   # Example accuracy of model 6
]

print(f'hasil model_1 = {model1}')
print(f'hasil model_2 = {model2}')
print(f'hasil model_3 = {model3}')
print(f'hasil model_4 = {model4}')
print(f'hasil model_5 = {model5}')
print(f'hasil model_6 = {model6}')

# Perform weighted majority voting
combined_predictions = weighted_majority_vote(predictions, accuracies)
print(f'Hasil Voting : {combined_predictions}')
