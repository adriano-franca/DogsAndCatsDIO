import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Caminho para o diretório contendo os dados descompactados
train_dir = '/content/dataset/archive.zip'

# Criando os geradores para processamento e augmentação de imagens
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Usar 20% para validação
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Criando os geradores de treinamento e validação
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training',
    follow_links=True  # Ignorar links ou arquivos quebrados
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Dividir os dados em conjunto de validação
)

# Construindo o modelo CNN
from tensorflow.keras.layers import Input
model = tf.keras.models.Sequential([
    Input(shape=(150, 150, 3)),  # Definir o tamanho da entrada aqui
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compilar o modelo
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Treinar o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=13,
    validation_data=validation_generator,
    validation_steps=50
)
# Salvar o modelo treinado
model.save('cat_dog_classifier.h5')