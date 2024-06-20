It is CNN + LSTM based model to generate caption for an image which is trained on Flicker 8000 Dataset and feature extraction from Xception Model using Transfer Learning.

About the implementation:

  1) Pre-processing:
     
         a) To preprocess the text data during training, I have used common NLP techniques like stemming, removing stopwords and used regex to remove punctuations.
     
         b) To preprocess images, I have used the cv2 module for resizing as Xception model is trained on images with dimensions 299x299.

     
  3) Tokenization:
         After preprocessing, I tokenized the text using keras.

     
  4) Model Designing:
     
         a) For the CNN part, feature extraction is done using Xception Model through Transfer Learning.
     
         b) For the LSTM part, after tokenization, I used pad_sequence to have same input length for all text descriptions.
     
         c) The model is trained in a way, that for predicting a word, the input is the prefix before that word and for the first word, it is '<'s'>'.
     
         d) Each text description is sandwiched between '<'s'>' and '<'e'>' to indentify the start and end of the sentence
     
         e) Note that the model is not a Sequential Neural Network model but a branching one, as there are 2 inputs i.e. the image and the prefix for a single word and since the dataset is very large, the entire data can not pe passed while training python generators are used while training the model that yields a batch of training data for the model to train.
     
  6) Finally, after training the model, I tested it on an out of sample image and it worked pretty good.
