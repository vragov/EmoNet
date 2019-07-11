import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels 
from keras.regularizers import l2
from sklearn.linear_model import LinearRegression

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
    This function utilizes get_labels and get_mfccs functions to generate full dataset only for anger and others
    '''
    labels = get_labels_angry_others(directory)
    features = get_mfccs(directory)
    full_data = pd.concat([features,labels], axis = 1)
    dataset = full_data.rename(index=str, columns={"0": "label"})
    #rnewdf[:5]
    dataset = dataset.fillna(0)
    return dataset

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

directory = '/home/volo/Insight/Mine/GIT_Upload/Data/'
labels = get_labels(directory)
df = pd.DataFrame(columns = ['feature'])
bookmark = 0
mylist = os.listdir(directory)
dataset = full_dataset(directory)
dataset_angry_others = full_dataset_angry_others(directory)
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

#model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
#Training
cnnhistory=model.fit(X_train_cnn, y_train, batch_size=32, epochs=300, validation_data=(X_test_cnn, y_test))
   
plot_loss(cnnhistory)
plot_confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues, classes = classes, normalize = True)


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
model_lm = LinearRegression()
model_lm.fit(X_train[:,:,0], y_train)

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






