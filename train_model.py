# train_model.py
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

dataset_dir = 'D:/Calories/dataset'

# Load calorie information from CSV
calorie_info_path = os.path.join(dataset_dir, 'calorie_info.csv')
calorie_info = pd.read_csv(calorie_info_path)

# Data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(dataset_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', classes=list(calorie_info['FoodItems']))

#CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(calorie_info), activation='softmax')) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10)  

# Saving
model.save('food_recognition_model.h5')
