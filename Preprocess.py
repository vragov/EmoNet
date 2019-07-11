import os
import scipy.io.wavfile
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras.layers import Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels 
from keras.regularizers import l2
from sklearn.linear_model import LinearRegression
import math

#Loading the model
def load_model(model_name):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("saved_models/"+model_name)
    loaded_model.compile(loss = 'categorical_crossentropy',
                         optimizer = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6),
                         metrics = ['accuracy'])
    return loaded_model

loaded_model = load_model(name)
score = loaded_model.evaluate(X_test_cnn, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


#Prediciton emotions on the test data
preds = loaded_model.predict(X_train_cnn, batch_size = 32, verbose = 1)
preds1 = preds.argmax(axis=1)
abc = preds1.astype(int).flatten()
predictions = (lb.inverse_transform((abc)))
preddf = pd.DataFrame({'predictedvalues':predictions})
actual = y_train.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((abc123)))
actualdf = pd.DataFrame({'actualvalues': actualvalues})
finaldf = actualdf.join(preddf)
finaldf.groupby('actualvalues').count()

finaldf.groupby('predictedvalues').count()

finaldf.to_csv('Predictions.csv', index=False)


# Test with my data 
data, sampling_rate = librosa.load('/home/volo/Insight/Mine/GIT_Upload/demo_customer_service.wav')

def get_features(file):
    X, sample_rate = librosa.load(file, res_type = 'kaiser_fast', 
                                  duration=3, sr=22050*2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 13),
                    axis = 0)
    featurelive = mfccs
    livedf2 = pd.DataFrame(data = featurelive).stack().to_frame().T
    twodim = np.expand_dims(livedf2, axis = 2)
    return twodim

waveform_features = get_features('/home/volo/Insight/Mine/GIT_Upload/demo_customer_service.wav')

def get_features_data(x,sr):
    sample_rate = np.array(sr)
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr = sample_rate, n_mfcc = 13),
                    axis = 0)
    featurelive = mfccs
    livedf2 = pd.DataFrame(data = featurelive).stack().to_frame().T
    twodim = np.expand_dims(livedf2, axis = 2)
    return twodim


file = '/home/volo/Insight/Mine/GIT_Upload/demo_customer_service.wav'
def generate_pred(file):
    data, sample_rate = librosa.load(file, res_type = 'kaiser_fast', sr=22050*2, offset=0.5)
    time_s = len(data)/sample_rate
    pred_time = []
    pred_proba_time = []
    for i in range(math.floor(time_s)-3):
        x = data[i*sample_rate:(i+3)*sample_rate]
        print(len(x))
        waveform_features = get_features_data(x,sample_rate)
        pred_proba = loaded_model.predict_proba(waveform_features, batch_size = 32, verbose = 1)
        pred = loaded_model.predict(waveform_features, batch_size = 32, verbose =1).argmax(axis = 1)
        livepred = lb.inverse_transform(pred.astype(int).flatten())
        #print(livepred)
        livepred = np.insert(livepred,0,i+2)
        pred_proba = np.insert(pred_proba[0],0,i+2)
        pred_time.append(livepred)
        pred_proba_time.append(pred_proba)
    return pred_time, pred_proba_time

pred_time_example, pred_proba_time = generate_pred(file)
labels_df = lb.classes_
labels_df = np.insert(labels_df, 0, 'time')
df = pd.DataFrame(pred_proba_time, columns = labels_df)
df.to_csv('/home/volo/Insight/Mine/GIT_Upload/foo.csv', header = True, index= False, sep = ',')

pred = loaded_model.predict(waveform_features, batch_size = 32, verbose =1).argmax(axis = 1)
livepred = lb.inverse_transform((pred.astype(int).flatten()))
livepred
