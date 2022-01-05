import numpy as np
import tensorflow as tf
import math

#npzのロード
path ='npm.npz'
with np.load(path) as data:
    landmarks = data['landmarks']
    label = data['label']

#データサイズ計測
data_size=math.floor(float(label.size)*0.8)

#訓練データとテストデータの分割
train_examples=landmarks[:data_size][:][:][:]
test_examples=landmarks[data_size:][:][:][:]
train_labels=label[:data_size][:][:][:]
test_labels=label[data_size:][:][:][:]

#データセット生成
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=((data_size,16,13,2), data_size)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)

model.evaluate(test_dataset)

