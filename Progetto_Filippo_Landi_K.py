# libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# usually the default matplotlib renderer
matplotlib.use("Agg") 
	
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-a", "--augment", type=int, default=-1,
	help="whether or not 'on the fly' data augmentation should be used")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-n", "--ncycle", type=int, default=1,
	help="number of cycles for the training")
args = vars(ap.parse_args())

# initialize batches, epochs and image resolution
INIT_LR = 1e-1
BS = 8
EPOCHS = 5
IMG_HEIGHT = 41
IMG_WIDTH = 19
 
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data_train = []
labels_train = []
data_test = [] 
labels_test = []
 
# loop over the image paths
for imagePath in imagePaths:
    # extract class label from directory of the image, load the image
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    # if image is a train examples
    if(imagePath.split(os.path.sep)[-3]=='Train'):
        # update the data and labels lists, respectively
        data_train.append(image)
        labels_train.append(label)
    # if image is a test example
    if(imagePath.split(os.path.sep)[-3]=='Test'):
        # update the data and labels lists, respectively
        data_test.append(image)
        labels_test.append(label)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data_train = np.array(data_train, dtype="float") / 255.0
data_test = np.array(data_test, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels_train = le.fit_transform(labels_train)
labels_train = np_utils.to_categorical(labels_train, 10)

labels_test = le.fit_transform(labels_test)
labels_test = np_utils.to_categorical(labels_test, 10)

# set the kfold with shuffle for better validation
kfold = KFold(shuffle=True)

# just a debug options
print("KFold setup: ")
print(kfold)
print("Train data and labels shape: ")
print(data_train.shape, labels_train.shape)

# initialize an our data augmenter as an "empty" image data generator
aug = ImageDataGenerator()

'''
In my opinion is a lot easier to use keras ImageDataGenerator to load a dataset,
because with one command you take everything you need: paths, images, labels and so on.
For kfold it was a problem because I couldn't properly divide the train data from
the validation with it because validation would remain static so not a cross validation.
For now it seems that the only route to do kfold is using sklearn+opencv and write lot more stuff.
'''

# check to see if we are applying "on the fly" data augmentation, and
# if so, re-instantiate the object
if args["augment"] > 0:
	print("[INFO] performing 'on the fly' data augmentation")
	aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		fill_mode="nearest"
                )

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# show network parameters and compile
model.summary()
model.compile(loss="categorical_crossentropy", optimizer='SGD', #o 'adam'
	metrics=["accuracy"])

script=input('Do you want to reload an already trained network and test? Y/n\n')
if script=='n':
    print('[INFO] training the network, it could take some time...')    
    # initialize lot of stuff
    # folded count collect the the overall X lenght when multiplied with epochs
    # folded name because it indicates times the dataset have been folded
    folded = 0
    # variables to collect over time to construct the overall graph
    tmp_loss=[]
    tmp_val_loss=[]
    tmp_train_acc=[]
    tmp_val_acc=[]

    # train the network
    for i in range(args['ncycle']):
        for train,test in kfold.split(data_train):
            # debug options to see kfold working properly
            # print(train.shape,test.shape)
            # print(train, test)
            print("[INFO] training 'folded network' for {} epochs...".format(EPOCHS))
            H = model.fit(
                aug.flow(data_train[train], labels_train[train], batch_size=BS
                         # debug to see the augmentation
                         # ,save_to_dir='AUG_D', save_prefix='AUG_D_', save_format='png'
                         ),
                # this is the crucial point that Image_Data_Generator for dataset loading
                # would not allow: I can apply it only to the train part and not the validation
                validation_data=(data_train[test], labels_train[test]),
                steps_per_epoch=len(data_train[train]) // BS,
                epochs=EPOCHS)
            # building overall history over folded and inner epochs
            tmp_loss = tmp_loss + H.history["loss"]
            tmp_val_loss = tmp_val_loss + H.history["val_loss"]
            tmp_train_acc = tmp_train_acc + H.history["accuracy"]
            tmp_val_acc= tmp_val_acc + H.history["val_accuracy"]
            folded=folded+1

    # lot of stuff to setup the png graph
    # N is the overall lenght of the X axis
    N = np.arange(0, EPOCHS*folded)
    # style is just the style, 'default', 'classic' etc. for other styles
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, tmp_loss, label="train_loss")
    plt.plot(N, tmp_val_loss, label="val_loss")
    plt.plot(N, tmp_train_acc, label="train_acc")
    plt.plot(N, tmp_val_acc, label="val_acc") 
    plt.xlabel("Epochs*Folded #")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    # saving the plot with the name given or with the default one
    plt.savefig(args["plot"])

    # save the model weights
    model.save_weights('./checkpoints/my_checkpoint')
    
if script == 'Y':
    model.load_weights('./checkpoints/my_checkpoint')
    
# evaluate the network on the test dataset, never seen before
print("[INFO] evaluating network on test data...")
predictions = model.predict(data_test, batch_size=BS)
print(classification_report(labels_test.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

