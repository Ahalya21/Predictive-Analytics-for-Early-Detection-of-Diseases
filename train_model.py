import tensorflow as tf

data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(224, 224),
    batch_size=32
)

class_names = data.class_names
print(class_names)


#Normalize data
data = data.map(lambda x,y: (x/255.0, y))


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(data, epochs=50)

model.save("eye_model.h5")
 