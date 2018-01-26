from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = \
    boston_housing.load_data()
# normalize the data, zero mean and 1 std
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
# always use the mean and std of training data
test_data -= mean
test_data /= std
# build the model
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
# validation using K-fold approach
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_history = []
# prepare the validation data from partition #k
for i in range(k):
    print('processing fold #',i)
    val_data = train_data[i * num_val_samples:(i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1)*num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_target = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_target,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_history.append(mae_history)

# compute the average of the per-epoch MAE scores of all folds
average_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]
# plot
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epoches')
plt.ylabel('Validation MAE')
plt.show()
# smoothe the curve and remove unrelated data point
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    if smoothed_points:
        previous = smoothed_points[-1]
        smoothed_points.append(previous * factor + points * (1 - factor))
    else:
        smoothed_points.append(points)
    return smoothed_points
# plot smoothed curve
plt.clf()
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()