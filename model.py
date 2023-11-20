import numpy as np
import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing

batch_size = 32
img_size = 512
channel = 3
epochs = 5


#load files from folder 
data = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/DL PROJECT/archive/Leather Defect Classification",
    shuffle = True,
    image_size = (img_size,img_size),
    batch_size = batch_size
)

classnames = data.class_names
print(classnames)
#Lets visualize the images

# for image,label in data.take(1):
#     for i in range(12):
#         ax = plt.subplot(3,4,i+1)
#         plt.title(classnames[label[i]])
#         plt.imshow(image[i].numpy().astype('uint8'))
#         plt.axis('off') 
# plt.show()

# Training Test and Validation split of Dataset 
def dataset_split(dataset,train_ratio = 0.7,test_ratio = 0.1,valid_ratio = 0.2,shuffle= True,shufflesize = 1000):
    assert (train_ratio+test_ratio+valid_ratio) == 1
    
    if shuffle:
        dataset = dataset.shuffle(shufflesize,seed = 12)
    
    data_size = len(dataset)
    train_size = int((data_size)*train_ratio)
    valid_size = int((data_size)*valid_ratio)

    train = data.take(train_size)
    valid = data.skip(train_size).take(valid_size)
    test  = data.skip(train_size).skip(valid_size) 

    return train,test,valid

train_data,test_data,valid_data = dataset_split(data)

print(f' --> Total images : {int(len(data))*32}')
print(f' --> Training set : {int(len(train_data))*32}')
print(f' --> Test set     : {int(len(test_data))*32}')
print(f' --> Validation set: {int(len(valid_data))*32}')

# Prefetch dataset 
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_data  = test_data.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
valid_data = valid_data.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

# Keras Sequential Layer 
resize_rescale = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(img_size,img_size),
        layers.experimental.preprocessing.Rescaling(1.0/255),
        layers.experimental.preprocessing.CenterCrop(256,256)
    ]
)

data_augment = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip(mode='horizontal_and_vertical'),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(-0.3,-0.2),
        layers.experimental.preprocessing.RandomContrast(0.8,0.1),
        

    ]
)
# Create Model (Model Architecture )
input_shape = (batch_size, img_size, img_size, channel)
classes = 6
model = models.Sequential(
    [
        resize_rescale,
        data_augment,
        
        layers.Conv2D(32,(3,3),activation='relu',input_shape= input_shape),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64,(3,3),activation='relu'),
        #layers.MaxPooling2D((2,2)),
        layers.Dropout(0.20),

        layers.Conv2D(128,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128,(3,3),activation='relu'),
        #layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.20),
        
        layers.Conv2D(128,(3,3),activation='relu'),
        #layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.20),
        layers.Dense(64,activation='relu'),
        layers.Dense(classes,activation='softmax')

    ]
)
model.build(input_shape)
model.summary()

# Model Compilation (Accuracy Metric)
model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)

# Fit the model
history = model.fit(
    train_data,
    batch_size = batch_size,
    validation_data = valid_data,
    verbose =1,
    epochs = epochs
    )

score = model.evaluate(test_data)
print(f'-->loss and accuracy:{score}')

loss = history.history['loss'] 
acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

# Plot the Loss and Accuracy

plt.subplot(1,2,1)
plt.plot(range(epochs),acc,label = 'training accuracy')
plt.plot(range(epochs),val_acc,label = 'validation accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(range(epochs),loss,label = 'training loss')
plt.plot(range(epochs),val_loss,label = 'validation loss')
plt.legend(loc = 'upper right')
plt.title('Training and validation Loss')
plt.show()

def predict(model,image):
    array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(array,axis=0)

    predict = model.predict(image_array)
    predict_class = classnames[np.argmax(predict[0])]
    conf = round(100 * (np.max(predict[0])),2)

    return predict_class,conf


# Make Prediction Now 

for imgbatch, labelbatch in test_data.take(1):
    for i in range(6):
        image = imgbatch[i].numpy().astype('uint8')
        actual = classnames[labelbatch[i]]
        
        predictedlabel,conf = predict(model,image)
        
        plt.subplot(2,3,i+1)
        plt.title(f'actual label:{actual}\npredicted label:{predictedlabel}\nconfidance:{conf}',fontsize = 8)
        plt.imshow(image)
        plt.axis('off') 
plt.show()