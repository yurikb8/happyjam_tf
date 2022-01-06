import numpy as np
import tensorflow as tf
import math

#npzのロード
path ='npm.npz' #npzが保存されてるパス
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

#バッチサイズ
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

#学習モデルの生成
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(16,13,2)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.trainable = True

#学習開始
model.fit(train_dataset, epochs=10)

#学習精度確認
model.evaluate(test_dataset)

#推論確認用
predict_result = model.predict(np.array([test_examples[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

# モデル全体を SavedModel として保存
model.save('saved_model/my_model')

#tflite変換
tflite_save_path = './gesture_classifier.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
open(tflite_save_path, 'wb').write(tflite_quantized_model)

