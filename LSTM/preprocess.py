from sklearn import preprocessing
import pandas as pd
import numpy as np
def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
    return df

def data_split(stock, seq_len,stock2):
    amount_of_features = len(stock.columns) # 5
    
    data = stock.as_matrix()
    data_same = stock2.as_matrix()
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 15days

    result = np.array(result)
    row = round(0.85 * result.shape[0]) # 85% split
    train = result[:int(row), :] # 85% date, all features 
    
    x_train = train[:, :-1] 
    y_train =np.array(train[:, -1][:,-1])
    
    x_test = result[int(row):, :-1] 
    y_test = np.array(result[int(row):, -1][:,-1])

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  
            
    result2 = []
    for index in range(len(data_same) - sequence_length): # maxmimum date = lastest date - sequence length
        result2.append(data_same[index: index + sequence_length]) # index : index + 15days
    result2 = np.array(result2)
    train2 = result2[:, :] # 85% date, all features 
    
    x_train2 = train2[:, :-1] 
    y_train2 =np.array(train2[:, -1][:,-1])

    x_train2 = np.reshape(x_train2, (x_train2.shape[0], x_train2.shape[1], amount_of_features))
    
    x_train=x_train.append(x_train2)
    y_train=y_train.append(y_train2)
    
    return [x_train, y_train, x_test, y_test]
