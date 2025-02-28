from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications, optimizers
import numpy as np

data_path = "J:/Book/"

N=500
ns=100
K=5

#Specify data generation model
mn=1      
nd=50
res_array = np.zeros( (ns,3))
learning_rate=0.01

#Specify output file name with model number and learning rate
fn_res = data_path + "m" + str(mn) + "_DL_" + str(learning_rate) + ".txt"

for i in range(ns):   
    data_path2 = "M:/Book/m" + str(mn) + "/"
    
    model = Sequential()
    model.add(Dense(nd, input_dim=5, activation='relu'))
    model.add(Dense(nd, activation='relu'))
    model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
    #learning_rate = .01
    opt = optimizers.Adam(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
    
    full_path = data_path2 + str(i+1) + ".csv"
    data = pd.read_csv(full_path)
    # Create Y_res and X_res
    Y_res = data.loc[data['r_idx'] == 1, 'Y']
    X_res = data.loc[data['r_idx'] == 1, data.columns[1:6]]

    # Create X and Y
    X = data.iloc[:, 1:6]
    Y = data['Y']
    rx= data['r_idx']
    
    # Create X_mis
    X_mis = data.loc[data['r_idx'] != 1, data.columns[1:6]]
    
    # Create rd DataFrame
    rd = pd.concat([Y_res, X_res], axis=1)
    
    # Create rdp DataFrame and transform 'r_idx' column
    rdp = data.iloc[:, 1:].copy()
    rdp['r_idx'] = rdp['r_idx'].map({0: 'N', 1: 'Y'})
    
    # Create folds
    kf = KFold(n_splits=K, shuffle=True, random_state=123)
    folds = list(kf.split(data['r_idx']))
    
    model.fit(
        X_res,
        Y_res,
        # batch_size = 32,
        validation_split = .25,
        epochs = 200,
        shuffle = True,
        callbacks = [es, mc],
        verbose = 0
    )
    # Call model
    saved_model = load_model('best_model.h5')
    # Extract Y hats from saved_model
    Yp = saved_model.predict(X_mis)
    Y_est_1=(sum(Y_res)+sum(Yp))/N

    model = Sequential()
    model.add(Dense(nd, input_dim=5, activation='relu'))
    model.add(Dense(nd, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    learning_rate = .001
    opt = optimizers.Adam(learning_rate = learning_rate)
    #opt = optimizers.Adam(learning_rate = learning_rate)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
    model.fit(
        X,
        rx,
        # batch_size = 32,
        validation_split = .25,
        epochs = 200,
        shuffle = True,
        callbacks = [es, mc],
        verbose = 0
    )
    # Call model
    saved_model = load_model('best_model.h5')
    # Extract Y hats from saved_model
    Rp = saved_model.predict(X)
    Rp = np.where(Rp == 0, 0.01, Rp)

    # Calculate Y_est_2
    Y_est_2 = np.sum(data['r_idx'] / Rp.squeeze() * data['Y']) / N

    # Calculate Y_est_3

    mu_estimates = np.zeros(K)  # Store estimates from each fold
    
    for k, (train_idx, test_idx) in enumerate(folds):
        # Define training and test sets
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        g_model_data = train_data[train_data['r_idx'] == 1].drop(columns=['r_idx'])
        e_model_data = train_data.drop(columns=['Y'])
        # Train nuisance functions
        model = Sequential()
        model.add(Dense(nd, input_dim=5, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))
        #learning_rate = .01
        opt = optimizers.Adam(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        model.fit(
            g_model_data.drop(columns='Y'),
            g_model_data['Y'],
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 0
        )
        # Call model
        saved_model = load_model('best_model.h5')
        
        g_X_test = saved_model.predict(test_data.drop(columns=['Y','r_idx']))


        # Propensity score model
        model = Sequential()
        model.add(Dense(nd, input_dim=5, activation='relu'))
        model.add(Dense(nd, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        learning_rate = .001
        opt = optimizers.Adam(learning_rate = learning_rate)
        #opt = optimizers.Adam(learning_rate = learning_rate)
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        model.fit(
            e_model_data.drop(columns='r_idx'),
            e_model_data['r_idx'],
            # batch_size = 32,
            validation_split = .25,
            epochs = 200,
            shuffle = True,
            callbacks = [es, mc],
            verbose = 0
        )
        # Call model
        saved_model = load_model('best_model.h5')
        # Extract Y hats from saved_model
        e_X_test = saved_model.predict(test_data.drop(columns=['Y','r_idx']))
        e_X_test = np.where(e_X_test == 0, 0.01, e_X_test)
        # Compute doubly robust estimate on test set
        mu_estimates[k] = np.mean(g_X_test.squeeze() + (test_data['r_idx'] / e_X_test.squeeze()) * (test_data['Y'] - g_X_test.squeeze()))
    
    # Compute final estimate (average over K folds)
    Y_est_3 = np.mean(mu_estimates)

    #np.savetxt(fn_res, bias_array)
    res_array[i,0]=Y_est_1
    res_array[i,1]=Y_est_2
    res_array[i,2]=Y_est_3

np.savetxt(fn_res, res_array, delimiter=',')
