from model_citra import model_1, model_2, model_3
from model_aroma import model_4, model_5, model_6


def weighted_average(values, weights):
    # Memastikan panjang values dan weights sama
    if len(values) != len(weights):
        raise ValueError("Panjang values dan weights harus sama")

    # Menghitung rata-rata berbobot
    numerator = sum(value * weight for value, weight in zip(values, weights))
    denominator = sum(weights)
    
    # Menghindari pembagian dengan nol
    if denominator == 0:
        raise ValueError("Jumlah dari weights tidak boleh nol")

    return numerator / denominator

def calculate_weights(accuracies):
    total_accuracy = sum(accuracies)
    weights = [accuracy / total_accuracy for accuracy in accuracies]
    return weights

accuracies = [
    0.73,  # Example accuracy of model 1
    0.75,  # Example accuracy of model 2
    0.73,  # Example accuracy of model 3
    1.0,  # Example accuracy of model 4
    1.0,  # Example accuracy of model 5
    0.94   # Example accuracy of model 6
]

nomor_matang = 17
nomor_mentah = 14
file = f'ripe/{nomor_matang} matang'
# file = f'unripe/{nomor_mentah} mentah'

model1 = model_1(f'data gambar/{file}.png')
model2 = model_2(f'data gambar/{file}.png')
model3 = model_3(f'data gambar/{file}.png')
model4 = model_4(f'data aroma/{file}.csv')
model5 = model_5(f'data aroma/{file}.csv')
model6 = model_6(f'data aroma/{file}.csv')



# Contoh penggunaan
values = [model1,model2,model3,model4,model5,model6]

weights = calculate_weights(accuracies)

weighted_avg = weighted_average(values, weights)

print(f'hasil model_1 = {model1}')
print(f'hasil model_2 = {model2}')
print(f'hasil model_3 = {model3}')
print(f'hasil model_4 = {model4}')
print(f'hasil model_5 = {model5}')
print(f'hasil model_6 = {model6}')

print("Weighted Average:", weighted_avg)