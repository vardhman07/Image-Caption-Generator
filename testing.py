from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
from keras.applications.xception import Xception # type: ignore
from keras.models import load_model # type: ignore
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

def extract_features(fname, model):
    try:
        image = cv2.imread(fname)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = cv2.resize(image,(299,299))
    # image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 's'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'e':
            break
    return in_text

max_length = 38
tokenizer = pickle.load(open("tokenizer.p","rb"))
model = load_model('models/model_9.keras')
xception_model = Xception(include_top=False, pooling="avg")

img_name = "3385593926_d3e9c21170.jpg"
photo = extract_features(img_name, xception_model)
img = cv2.imread(img_name)

description = generate_desc(model, tokenizer, photo, max_length)
print(description)
print("\n\n")
plt.imshow(img)