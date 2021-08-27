import tensorflow as tf

from tensorflow.keras import datasets,layers,models
from matplotlib import pyplot as plt


def plot_image(i, predictions_array, true_label, img):
    '''
    Plot test images and display the predicted and true label
    '''
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    '''
    Plot a bar chart of the softmax output with probabilites of different predicted labels
    '''
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

'''
Main function to load data from tensorflow database, create and train a
convolutional neural network model, and predict test images.
'''
if __name__=='__main__':
    # load data 
    #(train_images,train_labels),(test_images,test_labels)=datasets.cmaterdb.load_data()
    (train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()

    # normialize input images

    train_images,test_images=train_images/255,test_images/255

    # verify data

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10,10))

    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i+100])
        plt.xlabel(class_names[train_labels[i+100][0]])
    plt.show()

    # create convolutional model

    model=models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    # compile and train the model

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    #history = model.fit(train_images, train_labels, epochs=10, 
    #                   validation_data=(test_images, test_labels))

    model.fit(train_images, train_labels, epochs=10)

    # evaluate accuracy

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('Test Accuracy : ',test_acc)

    # create probability model to make predictions

    probability_model=tf.keras.Sequential([model,
                                           tf.keras.layers.Softmax()])
    predictions=probability_model.predict(test_images)

    # plot a few test images, their prediction and true labels.
    # Color code : green for correct prediction, red for wrong prediction

    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      plot_image(i, predictions[i], test_labels, test_images)
      plt.subplot(num_rows, 2*num_cols, 2*i+2)
      plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()
    plt.savefig("CIFAR10_predictions.jpg")
