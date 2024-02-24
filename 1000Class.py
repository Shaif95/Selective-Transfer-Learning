import pickle

# File path of the saved pickle
file_path = r"F:\combo_lists\whoi_combo_list.pickle"

# Load the data from the pickle file
with open(file_path, 'rb') as file:
    loaded_combo_list = pickle.load(file)

print("Combo list loaded successfully.")


from PIL import Image
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Define the dimensions for resizing
target_size = (32, 32)

# Initialize lists to store data
x_train = []
y_train = []

# Initialize label counter
label = 0

# Iterate through the directories
for directory in tqdm(loaded_combo_list):
    # Get the list of image files in the directory
    files = os.listdir(directory)

    # Iterate through the image files
    for file in files:
        # Check if the file is an image (you may want to add more checks)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load and resize the image
            img = Image.open(os.path.join(directory, file)).resize(target_size)

            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Convert image to numpy array and normalize
            img_array = np.array(img) / 255.0

            # Append image to x_train
            x_train.append(img_array)

            # Append label to y_train
            y_train.append(label)

    # Increment label for the next directory
    label += 1

# Convert lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=label)

# Verify the shapes
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Load the pre-trained MobileNet model (excluding the top layer)
#base_model = MobileNet(weights=None, include_top=False, input_shape=(32, 32, 3))
base_model = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))

# Add a global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer
x = Dense(512, activation='relu')(x)

# Add the output layer with the number of classes
output = Dense(y_train.shape[1], activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)


# In[ ]:


from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler

# Define a learning rate scheduler function
def lr_scheduler(epoch):
    # Set an initial learning rate
    initial_lr = 0.001
    
    # Define the decay factor
    decay_factor = 0.9
    
    # Calculate the new learning rate for this epoch
    new_lr = initial_lr * (decay_factor ** epoch)
    
    return new_lr

# Define the optimizer with SGD and momentum
optimizer = SGD(lr=0.001, momentum=0.9)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define the learning rate scheduler
lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model with the learning rate scheduler
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.1, callbacks=[lr_callback])


model.save('whoi_32_model.h5')