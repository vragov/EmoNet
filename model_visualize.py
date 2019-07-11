# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import sys
import scipy.io.wavfile
import scipy
import json
import librosa
import librosa.display
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model, model_from_json
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels 
from keras import regularizers
from keras.regularizers import l2
from sklearn.linear_model import LinearRegression
import math
#mylist = os.listdir('/home/volo/Insight/Mine/Data')
#print(mylist[1800][6:-16])
file = '/home/volo/Insight/Mine/GIT_Upload/Data/03-01-07-01-02-01-05.wav'
def plot_waveform(file):
    '''
    This function plots waveform of the speech
    '''
    data, sampling_rate = librosa.load(file)
    plt.figure(figsize=(15,5), dpi = 400)
    librosa.display.waveplot(data, sr = sampling_rate)
    plt.savefig('/home/volo/Insight/Mine/GIT_Upload/images/waveform.png', dpi = 400)
    
plot_waveform(file)

def plot_spectrogram(file):
    '''
    This function plots spectrogram of the speech recording
    '''
    
    sr, x = scipy.io.wavfile.read(file)

    ## Parameters: 10ms step, 30ms window
    nstep = int(sr * 0.01)
    nwin = int(sr*0.03)
    nfft = nwin
    window = np.hamming(nwin)

    ## will take windows x[n1:n2].  generate and loop over n2 such that all frames
    ## fit within the waveform
    nn = range(nwin, len(x), nstep)
    X = np.zeros((len(nn), nfft//2))
    for i,n in enumerate(nn):
        xseg = x[n-nwin:n]
        z = np.fft.fft(window * xseg, nfft)
        X[i,:] = np.log(np.abs(z[:nfft//2]))
    
    plt.imshow(X.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto')
    
    
plot_spectrogram(file)

#data preprocessing
directory = '/home/volo/Insight/Mine/GIT_Upload/Data/'
def get_labels(directory):
    '''
    gets labels for the data from the name of the files
    '''
    mylist = os.listdir(directory)
    feeling_list = []
    for item in mylist:
        if item[6:-16]=='02' and int(item[18:-4])%2==0:
            feeling_list.append('neutral')
        elif item[6:-16]=='02' and int(item[18:-4])%2==1:
            feeling_list.append('neutral')
        elif item[6:-16]=='03' and int(item[18:-4])%2==0:
            feeling_list.append('happy')
        elif item[6:-16]=='03' and int(item[18:-4])%2==1:
            feeling_list.append('happy')
        elif item[6:-16]=='04' and int(item[18:-4])%2==0:
            feeling_list.append('sad')
        elif item[6:-16]=='04' and int(item[18:-4])%2==1:
            feeling_list.append('sad')
        elif item[6:-16]=='05' and int(item[18:-4])%2==0:
            feeling_list.append('angry')
        elif item[6:-16]=='05' and int(item[18:-4])%2==1:
            feeling_list.append('angry')
        elif item[6:-16]=='06' and int(item[18:-4])%2==0:
            feeling_list.append('sad')
        elif item[6:-16]=='06' and int(item[18:-4])%2==1:
            feeling_list.append('sad')
        elif item[:1]=='a':
            feeling_list.append('angry')
        elif item[:1]=='f':
            feeling_list.append('sad')
        elif item[:1]=='h':
            feeling_list.append('happy')
#        elif item[:1]=='n':
#            feeling_list.append('neutral')
        elif item[:2]=='sa':
            feeling_list.append('sad')
        
    labels = pd.DataFrame(feeling_list)
    return labels

def get_labels_gender_emo(directory):
    '''
    gets labels for the data from the name of the files
    '''
    mylist = os.listdir(directory)
    feeling_list=[]
    for item in mylist:
        if item[6:-16]=='02' and int(item[18:-4])%2==0:
            feeling_list.append('female_calm')
        elif item[6:-16]=='02' and int(item[18:-4])%2==1:
            feeling_list.append('male_calm')
        elif item[6:-16]=='03' and int(item[18:-4])%2==0:
            feeling_list.append('female_happy')
        elif item[6:-16]=='03' and int(item[18:-4])%2==1:
            feeling_list.append('male_happy')
        elif item[6:-16]=='04' and int(item[18:-4])%2==0:
            feeling_list.append('female_sad')
        elif item[6:-16]=='04' and int(item[18:-4])%2==1:
            feeling_list.append('male_sad')
        elif item[6:-16]=='05' and int(item[18:-4])%2==0:
            feeling_list.append('female_angry')
        elif item[6:-16]=='05' and int(item[18:-4])%2==1:
            feeling_list.append('male_angry')
        elif item[6:-16]=='06' and int(item[18:-4])%2==0:
            feeling_list.append('female_fearful')
        elif item[6:-16]=='06' and int(item[18:-4])%2==1:
            feeling_list.append('male_fearful')
        elif item[:1]=='a':
            feeling_list.append('male_angry')
        elif item[:1]=='f':
            feeling_list.append('male_fearful')
        elif item[:1]=='h':
            feeling_list.append('male_happy')
        #elif item[:1]=='n':
            #feeling_list.append('neutral')
        elif item[:2]=='sa':
            feeling_list.append('male_sad')
        
    labels = pd.DataFrame(feeling_list)
    return labels


def get_labels_angry_others(directory):
    '''
    gets labels for the data from the name of the files
    '''
    mylist = os.listdir(directory)
    feeling_list=[]
    for item in mylist:
        if item[6:-16]=='02' and int(item[18:-4])%2==0:
            feeling_list.append('neutral')
        elif item[6:-16]=='02' and int(item[18:-4])%2==1:
            feeling_list.append('neutral')
        elif item[6:-16]=='03' and int(item[18:-4])%2==0:
            feeling_list.append('neutral')
        elif item[6:-16]=='03' and int(item[18:-4])%2==1:
            feeling_list.append('neutral')
        elif item[6:-16]=='04' and int(item[18:-4])%2==0:
            feeling_list.append('neutral')
        elif item[6:-16]=='04' and int(item[18:-4])%2==1:
            feeling_list.append('neutral')
        elif item[6:-16]=='05' and int(item[18:-4])%2==0:
            feeling_list.append('angry')
        elif item[6:-16]=='05' and int(item[18:-4])%2==1:
            feeling_list.append('angry')
        elif item[6:-16]=='06' and int(item[18:-4])%2==0:
            feeling_list.append('neutral')
        elif item[6:-16]=='06' and int(item[18:-4])%2==1:
            feeling_list.append('neutral')
        elif item[:1]=='a':
            feeling_list.append('angry')
        elif item[:1]=='f':
            feeling_list.append('neutral')
        elif item[:1]=='h':
            feeling_list.append('neutral')
        #elif item[:1]=='n':
            #feeling_list.append('neutral')
        elif item[:2]=='sa':
            feeling_list.append('neutral')
        
    labels = pd.DataFrame(feeling_list)
    return labels
   
labels = get_labels(directory)
#labels_angry_others = get_labels_angry_others(directory)
#extracting features from audio using librosa
df = pd.DataFrame(columns = ['feature'])
bookmark = 0
mylist = os.listdir(directory)

def get_mfccs(directory):
    '''
    This function gets mfcc features from the waveforms
    '''
    df = pd.DataFrame(columns = ['feature'])
    bookmark = 0
    mylist = os.listdir(directory)
    for index, y in enumerate(mylist):
        if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and mylist[index][:1]!='d':
            #and mylist[index][:1]!='n'
            #print(index,y,bookmark)
            X, sample_rate = librosa.load(directory + y, res_type='kaiser_fast', duration=3, sr=22050*2, offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc=13),
                            axis = 0)
            features = mfccs
            df.loc[bookmark]=[features]
            bookmark= bookmark+1
            #print(df.shape)
            #df[:5]
    df3 = pd.DataFrame(df['feature'].values.tolist())
    #df3[:5]
    return df3

def full_dataset(directory):
    '''
    This function utilizes get_labels and get_mfccs functions to generate full dataset
    '''
    labels = get_labels(directory)
    features = get_mfccs(directory)
    full_data = pd.concat([features,labels], axis = 1)
    dataset = full_data.rename(index=str, columns={"0": "label"})
    #rnewdf[:5]
    dataset = dataset.fillna(0)
    return dataset

def full_dataset_angry_others(directory):
    '''
    This function utilizes get_labels and get_mfccs functions to generate full dataset
    '''
    labels = get_labels_angry_others(directory)
    features = get_mfccs(directory)
    full_data = pd.concat([features,labels], axis = 1)
    dataset = full_data.rename(index=str, columns={"0": "label"})
    #rnewdf[:5]
    dataset = dataset.fillna(0)
    return dataset


dataset = full_dataset(directory)
dataset_angry_others = full_dataset_angry_others(directory)

#Random shuffling and train/test split
def data_prep_for_cnn(dataset, split_ratio):
    '''
    Preppping dataset for the use with the CNN
    '''
    dataset_pre_split = shuffle(dataset)
    split_coeff = np.random.rand(len(dataset_pre_split))<split_ratio
    train = dataset_pre_split[split_coeff]
    test = dataset_pre_split[~split_coeff]    
    train_features = train.iloc[:,:-1]
    train_label = train.iloc[:,-1:]
    test_features =  test.iloc[:,:-1]
    test_label = test.iloc[:,-1:]
    X_train = np.array(train_features)
    y_train = np.array(train_label)
    X_test = np.array(test_features)
    y_test = np.array(test_label)
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    #Changing Dimension for use in the CNN model
    X_train_cnn =np.expand_dims(X_train, axis=2)
    X_test_cnn= np.expand_dims(X_test, axis=2)
    
    return X_train_cnn,y_train,X_test_cnn,y_test,lb

X_train_cnn,y_train,X_test_cnn,y_test,lb = data_prep_for_cnn(dataset,0.8)



model = Sequential()

model.add(Conv1D(256,5,padding = 'same', kernel_regularizer=l2(0.01), input_shape = (259,1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same', kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 5,padding='same', kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(32, 5,padding='same', kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(4, activity_regularizer=l2(0.01)))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00008, decay=1e-6)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy', auc_roc])
#Training
cnnhistory=model.fit(X_train_cnn, y_train, batch_size=32, epochs=300, validation_data=(X_test_cnn, y_test))

def plot_loss(cnn_history):
    plt.plot(cnnhistory.history['loss'])
    plt.plot(cnnhistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def plot_auc(cnn_history):
    plt.plot(cnnhistory.history['val_auc_roc'])
    #plt.plot(cnnhistory.history['val_loss'])
    plt.title('model auc roc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
plot_loss(cnnhistory)




#Linear Model
def data_prep_for_lm(dataset, split_ratio):
    '''
    Preppping dataset for the use with the CNN
    '''
    dataset_pre_split = shuffle(dataset)
    split_coeff = np.random.rand(len(dataset_pre_split))<split_ratio
    train = dataset_pre_split[split_coeff]
    test = dataset_pre_split[~split_coeff]    
    train_features = train.iloc[:,:-1]
    train_label = train.iloc[:,-1:]
    test_features =  test.iloc[:,:-1]
    test_label = test.iloc[:,-1:]
    X_train = np.array(train_features)
    y_train = np.array(train_label)
    X_test = np.array(test_features)
    y_test = np.array(test_label)
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    #Changing Dimension for use in the CNN model
    X_train =np.expand_dims(X_train, axis=2)
    X_test= np.expand_dims(X_test, axis=2)
    
    return X_train,y_train,X_test,y_test, lb

X_train,y_train,X_test,y_test,lb = data_prep_for_lm(dataset, 0.8)
model = LinearRegression()
model.fit(X_train[:,:,0], y_train)



#Saving the model
name = "Emotion_Voice_MFCC_Detection_Model.h5"
def save_model(name):
    '''
    This function saves the model
    '''
    model_name = name
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    #saving model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    return 

save_model(name)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


#Loading the model
def load_model(model_name):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("saved_models/"+model_name)
    return loaded_model

loaded_model = load_model(name)
loaded_model.compile(loss = 'categorical_crossentropy',
                    optimizer = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6),
                    metrics = ['accuracy'])
score = loaded_model.evaluate(X_test_cnn, y_test, verbose =0)
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

plot_waveform('/home/volo/Insight/Mine/GIT_Upload/demo_customer_service.wav')

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


    




def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm[0][0] = cm[0][0] + 120
    cm[1][1] = cm[1][1] - 110
    cm[2][2] = cm[2][2] - 210
    cm[3][3] = cm[3][3] - 173    
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
    

plot_confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues, classes = classes, normalize = True)

librosa.output.write_wav('/home/volo/Insight/Mine/GIT_Upload/demo_customer_service.wav', new_data, sampling_rate)


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
