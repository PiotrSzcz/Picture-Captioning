import os
import nltk
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.utils import load_img, img_to_array
from keras.applications.xception import Xception
from nltk.translate.bleu_score import corpus_bleu
from keras.utils import to_categorical, plot_model
from nltk.translate.meteor_score import meteor_score
from keras_preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.densenet import DenseNet121, DenseNet201
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

BASE_DIR = 'flickr8k'
F30k_DIR = 'flickr30k'
WORKING_DIR = 'working'

def load_backbone():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model

def getOutputShapeBackBone(model):
    print("-------- Testing shape of en loaded Backbone structure -------")
    input_vector = tf.random.normal((1, *model.input_shape[1:]))
    output_vector = model.predict(input_vector)
    vector_length = len(output_vector.flatten())
    return vector_length

def load_features_from_file(file_path, base_dir, working_dir, model):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            loaded_features = pickle.load(file)
        return loaded_features
    else:
        print(f"---------- File {file_path} does not exist. Trying to recreate -----------")
        pickle_me_baby(base_dir, working_dir, model)
        return load_features_from_file(file_path, base_dir, working_dir, model)

def pickle_me_baby(base_dir, working_dir, model):
    idx = 0
    features = {}
    directory = os.path.join(base_dir, 'Images')
    dirSize = len(os.listdir(directory))
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature
        idx += 1
        print(f'Feature extraction progress: {idx}/{dirSize}')
    pickle.dump(features, open(os.path.join(working_dir, 'features.pkl'), 'wb'))


def load_captions(file_path):
    with open(file_path, 'r') as f:
        next(f)
        captions_doc = f.read()
    return captions_doc

def create_captions_mapping(captions_doc):
    mapping = {}
    captions_lines = captions_doc.split('\n')
    for idx, line in enumerate(captions_lines):
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping

def clean_captions(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption

def get_all_captions(mapping):
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
    return all_captions

def create_tokenizer(all_captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

def get_model_inputs_outputs(inputShape, vocab_size, max_length):
    inputs1 = Input(shape=(inputShape,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    return inputs1, inputs2, outputs

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0

def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam')

def train_model(epochs, batch_size, model, features, mapping, tokenizer, vocab_size, max_length, train, data_generator):
    steps = len(train) // batch_size
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.05, patience=3, restore_best_weights=True)
    for i in range(epochs):
        generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1, callbacks=[early_stopping])
    model.save(os.path.join(WORKING_DIR, 'best_model.h5'))

def fine_tune_model(epochs, batch_size, model, features, mapping, tokenizer, vocab_size, max_length, train, data_generator):
    print("------ Starting proces of retraining model  -------")
    steps = len(train) // batch_size
    model.load_weights(os.path.join(WORKING_DIR, 'best_model.h5'))
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.05, patience=3, restore_best_weights=True)
    for i in range(epochs):
        generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1, callbacks=[early_stopping])
    model.save(os.path.join(WORKING_DIR, 'best_model_fine_tuned.h5'))

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, features, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

def generate_caption(model, features, tokenizer, max_length, image_name, mapping):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print(f'--------Predicting caption for image {image_name}---------')
    print('--------------------- Actual ---------------------')
    for caption in captions:
        print(caption)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('-------------------- Predicted --------------------')
    print(y_pred)
    plt.imshow(image)

def predictValidationData(image_path, model, tokenizer, max_length):
    vgg_model = load_backbone()
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    y=predict_caption(model, feature, tokenizer, max_length)
    print(f'--------Predicting caption for image {image_path}---------')
    print(y)

def evaluate_model(model, features, tokenizer, max_length, test, mapping):
    print(f"--------- Starting proces of evalueating model ----------")
    actual, predicted = list(), list()
    total_images = len(test)
    meteor_scores = []
    for idx, key in enumerate(test):
        captions = mapping[key]
        y_pred = predict_caption(model, features[key], tokenizer, max_length)
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        actual.append(actual_captions)
        predicted.append(y_pred)
        bleu1_score = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
        bleu2_score = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        bleu3_score = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
        bleu4_score = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
        meteor = meteor_score(actual_captions, y_pred, alpha=0.9, beta=3.0, gamma=0.5)
        meteor_scores.append(meteor)
        m_score = np.mean(meteor_scores)
        print(f"Progress: {idx + 1}/{total_images} - BLEU-1: {bleu1_score:.4f} - BLEU-2: {bleu2_score:.4f} - BLEU-3: {bleu3_score:.4f} - BLEU-4: {bleu4_score:.4f} - METEOR: {m_score:.4f}")

def prepare_data(model):
    features_file_path = os.path.join(WORKING_DIR, 'features.pkl')
    features = load_features_from_file(features_file_path, BASE_DIR, WORKING_DIR, model)
    captions_file_path = os.path.join(BASE_DIR, 'captions.txt')
    captions_doc = load_captions(captions_file_path)
    mapping = create_captions_mapping(captions_doc)
    clean_captions(mapping)
    all_captions = get_all_captions(mapping)
    tokenizer = create_tokenizer(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in all_captions)
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    train = image_ids[:split]
    test = image_ids[split:]
    return features, mapping, tokenizer, vocab_size, max_length, train, test

def predict_and_evaluate_on_dataset(new_image_dir, model, tokenizer, max_length, mapping):
    print(f"--------- Testing trained model on enother dataset from {new_image_dir} ----------")
    backbone_model = load_backbone()
    actual, predicted = list(), list()
    total_images = len(os.listdir(new_image_dir))
    testSize = int(total_images*0.05)
    meteor_scores = []
    imageList = os.listdir(new_image_dir)[total_images-testSize:]
    for idx, img_name in enumerate(imageList):
        img_path = os.path.join(new_image_dir, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = backbone_model.predict(image, verbose=0)
        predicted_caption = predict_caption(model, feature, tokenizer, max_length)
        image_id = img_name.split('.')[0]
        true_captions = mapping.get(image_id, [])
        predicted_caption_tokens = predicted_caption.split()
        true_captions_tokens = [caption.split() for caption in true_captions]
        actual.append(true_captions_tokens)
        predicted.append(predicted_caption_tokens)
        bleu1_score = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
        bleu2_score = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        bleu3_score = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
        bleu4_score = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
        meteor = meteor_score(true_captions_tokens, predicted_caption_tokens, alpha=0.9, beta=3.0, gamma=0.5)
        meteor_scores.append(meteor)
        avg_meteor = np.mean(meteor_scores)
        print(f"Progress: {idx + 1}/{testSize} - BLEU-1: {bleu1_score:.4f} - BLEU-2: {bleu2_score:.4f} - BLEU-3: {bleu3_score:.4f} - BLEU-4: {bleu4_score:.4f} - METEOR: {avg_meteor:.4f}")
        
def main():
    nltk.download('wordnet')
    model = load_backbone()
    lstmInputShape = getOutputShapeBackBone(model)

    features, mapping, tokenizer, vocab_size, max_length, train, test = prepare_data(model)

    inputs1, inputs2, outputs = get_model_inputs_outputs(lstmInputShape, vocab_size, max_length)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    compile_model(model)

    if not os.path.exists(os.path.join(WORKING_DIR, 'best_model.h5')):
        print("------ Starting proces of training model  -------")
        train_model(10, 64, model, features, mapping, tokenizer, vocab_size, max_length, train, data_generator)
    else:
        model = load_model(os.path.join(WORKING_DIR, 'best_model.h5'), compile=False)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    evaluate_model(model, features, tokenizer, max_length, train, mapping) 
    #generate_caption(model, features, tokenizer, max_length, "241346471_c756a8f139.jpg", mapping)
    #predictValidationData('C:/Users/Piotr/Downloads/1.jpg', model, tokenizer, max_length)
    #fine_tune_model(1, 64, model, features, mapping, tokenizer, vocab_size, max_length, train, data_generator)
    #predict_and_evaluate_on_dataset(F30k_DIR + '/Images', model, tokenizer, max_length, create_captions_mapping(load_captions(os.path.join(F30k_DIR, 'captions.txt'))))

if __name__ == "__main__":
    main()