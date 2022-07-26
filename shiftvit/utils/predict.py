import numpy as np 
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import from_pretrained_keras
from .custom_objects import WarmUpCosine
from .constants import Config, class_vocab
from keras.utils import load_img, img_to_array
from tensorflow_addons.optimizers import AdamW
import matplotlib.pyplot as plt
import pandas as pd
import random
config = Config()

##Load Model
model = from_pretrained_keras("shivi/shiftvit", custom_objects={"WarmUpCosine":WarmUpCosine, "AdamW": AdamW})

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


AUTO = tf.data.AUTOTUNE

def predict(image_path):
    """
    This function is used for fetching predictions corresponding to input_dataframe.
    It outputs another dataframe containing: 
        1. prediction probability for each class
        2. actual expected outcome for each entry in the input dataframe
    """
    
    test_image1 = load_img(image_path,target_size =(32,32))
    test_image = img_to_array(test_image1) 
    test_image = np.expand_dims(test_image, axis =0)
    test_image = test_image.astype('uint8')
    
    
    predict_ds = tf.data.Dataset.from_tensor_slices(test_image)
    predict_ds = predict_ds.shuffle(config.buffer_size).batch(config.batch_size).prefetch(AUTO)
    logits = model.predict(predict_ds) 
    prob = tf.nn.softmax(logits)

    confidences = {}
    prob_list = prob.numpy().flatten().tolist()
    sorted_prob = np.argsort(prob)[::-1].flatten()
    for i in sorted_prob:
        confidences[class_vocab[i]] = float(prob_list[i])
    
    return confidences


def predict_batch(image_path):
   
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(config.batch_size).prefetch(AUTO)
    #test_ds = test_ds.shuffle(100)
    slice = test_ds.take(1)

    # img_gen = tf.keras.preprocessing.image.ImageDataGenerator()

    # images, _ = next(img_gen.flow_from_directory("examples/set1"))

    # ds = tf.data.Dataset.from_generator(
    #     lambda: img_gen.flow_from_directory("examples",target_size=(32, 32)), 
    # output_types=(tf.uint8, tf.uint8), 
    # output_shapes=([None,32,32,3], [None,4]))
    # slice = ds.take(1)
    
    slice_pred = model.predict(slice)
    slice_pred = tf.nn.softmax(slice_pred)
    
    saved_plot = "plot.jpg"
    fig = plt.figure()
    
    predictions_df = pd.DataFrame()
    num =  random.randint(0,50)
    for images, labels in slice:
      for i,j in zip(range(num,num+3), range(3)):
            ax = plt.subplot(1, 3, j + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            output = np.argmax(slice_pred[i])

            prob_list = slice_pred[i].numpy().flatten().tolist()
            sorted_prob = np.argsort(slice_pred[i])[::-1].flatten()
            prob_scores = {"image": "image "+ str(j), "1st_highest_probability": f"prob of {class_vocab[sorted_prob[0]]} is {round(prob_list[sorted_prob[0]] * 100,2)} %", 
            "2nd_highest_probability": f"prob of {class_vocab[sorted_prob[1]]} is {round(prob_list[sorted_prob[1]] * 100,2)} %", 
            "3rd_highest_probability": f"prob of {class_vocab[sorted_prob[2]]} is {round(prob_list[sorted_prob[2]] * 100,2)} %"}
            predictions_df = predictions_df.append(prob_scores,ignore_index=True)
            
            plt.title(f"image {j} : {class_vocab[output]}")
            plt.axis("off")
            plt.savefig(saved_plot,bbox_inches='tight')

    return saved_plot, predictions_df







