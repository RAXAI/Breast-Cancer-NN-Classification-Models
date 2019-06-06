import pandas as pd
import numpy as np
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools
#fix standard seed
seed = 21
np.random.seed(seed)

df = pd.read_csv('dataR2.csv')
# change class values  to 1 and 0 makes more since since
#we'll be using a sigmoid function for the final layer generates an
#output between 1 and 0
df['Classification'] = df['Classification' ].apply(lambda x: x-1)
X = np.array(df.drop(['Classification'],axis=1))
y=  np.array(df['Classification']).reshape(-1,1)
n_cols = X.shape[1]
#appy standard scaler to all features
X = StandardScaler().fit_transform(X)
#make train test split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=seed,stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=seed,stratify=y_train)

#pca = PCA(n_components=6)
#pca.fit(X)
#X =pca.fit_transform(X)

'''
def plot_history(history): 
    # use for loop make pass in a list of different history objects for different models
    for x in history
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


'''



'''

# specified model
def model1():
    model = Sequential()
    model.add(Dense(9,kernel_initializer='uniform',input_shape = (n_cols,),activation= 'relu'))
    #model.add(Dense(5,kernel_initializer='uniform', activation= 'relu'))
    #model.add(Dense(3,kernel_initializer='uniform', activation= 'relu'))
    model.add(Dense(1,kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
def model2():
    model = Sequential()
    model.add(Dense(9,kernel_initializer='uniform',input_shape = (n_cols,),activation= 'relu'))
    model.add(Dense(5,kernel_initializer='uniform', activation= 'relu'))
    model.add(Dense(1,kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
'''
def model3(optimizer='adam', init='uniform'):
    model = Sequential()
    model.add(Dense(4,kernel_initializer=init,input_shape = (n_cols,),activation= 'relu'))
    #model.add(Dense(4,kernel_initializer='uniform', activation= 'relu'))
    #model.add(Dense(2,kernel_initializer='uniform', activation= 'relu'))
    model.add(Dense(1,kernel_initializer=init, activation='sigmoid'))
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return model
'''
def model4():
    model = Sequential()
    model.add(Dense(9,kernel_initializer='uniform',input_shape = (n_cols,),activation= 'relu'))
    model.add(Dense(6,kernel_initializer='uniform', activation= 'relu'))
    model.add(Dense(3,kernel_initializer='uniform', activation= 'relu'))
    model.add(Dense(1,kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
'''
#compile model 0.782609 (0.062280) with: {'batch_size': 5, 'epochs': 200, 'init': 'glorot_uniform', 'optimizer': 'adam'}

#estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=10, verbose=0)
#history = model1().fit(X_train,y_train, validation_data=(X_val,y_val),batch_size=32,verbose=0,epochs=150)
#history2 = model2().fit(X_train,y_train, validation_data=(X_val,y_val),batch_size=32,verbose=0,epochs=150)
model3().fit(X_train,y_train,batch_size=32,verbose=0,epochs=1000)
#history4 = model3().fit(X_train,y_train, validation_data=(X_val,y_val),batch_size=32,verbose=0,epochs=150)

'''
plt.figure(1)
plt.plot(history.history['acc'], 'b', label='Training accuracy (' + str(format(history.history['acc'][-1],'.5f'))+')')
plt.plot(history.history['val_acc'], 'g', label='Validation accuracy (' + str(format(history.history['val_acc'][-1],'.5f'))+')')
plt.title('model 1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()

plt.figure(2)
plt.plot(history2.history['acc'], 'b', label='Training accuracy (' + str(format(history2.history['acc'][-1],'.5f'))+')')
plt.plot(history2.history['val_acc'], 'g', label='Validation accuracy (' + str(format(history2.history['val_acc'][-1],'.5f'))+')')
plt.title('model 2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()

'''
'''
plt.figure(3)
plt.plot(history3.history['acc'], 'b', label='Training accuracy (' + str(format(history3.history['acc'][-1],'.5f'))+')')
plt.plot(history3.history['val_acc'], 'g', label='Validation accuracy (' + str(format(history3.history['val_acc'][-1],'.5f'))+')')
plt.title('model 3 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
'''

'''
plt.figure(4)
plt.plot(history4.history['acc'], 'b', label='Training accuracy (' + str(format(history4.history['acc'][-1],'.5f'))+')')
plt.plot(history4.history['val_acc'], 'g', label='Validation accuracy (' + str(format(history4.history['val_acc'][-1],'.5f'))+')')
plt.title('model 4 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend() 
plt.show()'''
'''

#model1 = KerasClassifier(build_fn=model1, epochs=150, batch_size=32, verbose=0)
#model2 = KerasClassifier(build_fn=model2, epochs=150, batch_size=32, verbose=0)
 model3 = KerasClassifier(build_fn=model3,verbose=0)
#model4 = KerasClassifier(build_fn=model4, epochs=150, batch_size=32, verbose=0)

'''

model3a = KerasClassifier(build_fn=model3,verbose=0,epochs=1000,batch_size=32)

model3a.fit(X_train,y_train)
probs = model3a.predict_proba(X_test)
probs = probs[:,1]
auc = roc_auc_score(y_test,probs)
print('AUC: %.3f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.title('Model 3 ROC graph')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# show the plot
plt.show()

# summarize history for loss
'''plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
#score1 = model1().evaluate(X_test, y_test, batch_size=32)
#score2 = model1().evaluate(X_test, y_test, batch_size=32)
#score3 = model3().evaluate(X_test, y_test, batch_size=32)
#score4 = model1().evaluate(X_test, y_test, batch_size=32)
#print(score3)

'''
optimizers = ['rmsprop', 'adam','sgd',]
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150, 200]
batches = [5, 10, 20, 32]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model3, param_grid=param_grid,cv=5)
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix model 3 but with 1000 epochs'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
'''    
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           binary=True):

    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x)
    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5))    
    
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,classes=classes)

#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, X_train,y_train, cv = kfold)
#print(results)
#print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#acurracy plot 
#score = model.evaluate(X_test, y_test)
#print('Score: ',score[1]*100)
#plot_history(history)
#plot_history(history)
full_multiclass_report(model3(),X_test,y_test,np.arange(2))
'''
