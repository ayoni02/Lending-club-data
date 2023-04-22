import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
sns.set_style('darkgrid')
#df = pd.read_csv('lending_club_loan_two.csv')

def opendata(a):
    df = pd.read_csv(a)
    return df

def info(df):
    print(df.info())
    print(df.describe())
    print(df.head())

def visualization(df):
    sns.countplot(x='loan_status',data=df)
    plt.show()
    sns.histplot(df['loan_amnt'],kde=None,bins=30)
    plt.show()
    sns.heatmap(df.corr(),annot=True,cmap='viridis')
    plt.show()
    sns.boxplot(x='loan_status',y='loan_amnt',data=df)
    plt.show()
    ax = sns.countplot(x='purpose',data=df)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=40, ha='right')
    #plt.tight_layout()
    plt.show()
    ax = sns.countplot(x='purpose',hue='loan_status',data=df)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=40, ha='right')
    plt.show()
    sns.countplot(x='grade',data=df,hue='loan_status',order=sorted(df['grade'].unique()))
    plt.show()
    #plt.figure(figsize=(12,6))
    sns.scatterplot(x='installment',y='loan_amnt',data=df)
    plt.show()


def drop(df):
    df.drop(['emp_length','emp_title','title','grade'],axis=1,inplace=True)
    total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
    def fillup(q,w):
        if np.isnan(w):
            return total_acc_avg[q]
        else:
            return w
    df['mort_acc'] = df.apply(lambda y: fillup(y['total_acc'], y['mort_acc']), axis=1)
    df.dropna(inplace=True)
    return df
    

def feature(df):
    df['loan_repaid'] = pd.get_dummies(df['loan_status'],drop_first=True)
    suba = pd.get_dummies(df['sub_grade'],drop_first=True)
    df = pd.concat([df,suba],axis=1)
    df.drop('sub_grade',axis=1,inplace=True)
    df['term'] = df['term'].apply(lambda x: int(x[:3]))
    subb = pd.get_dummies(df['verification_status'],drop_first=True)
    df = pd.concat([df,subb],axis=1)
    subc = pd.get_dummies(df['application_type'],drop_first=True)
    df = pd.concat([df,subc],axis=1)
    subd = pd.get_dummies(df['initial_list_status'],drop_first=True)
    df = pd.concat([df,subd],axis=1)
    subd = pd.get_dummies(df['purpose'],drop_first=True)
    df = pd.concat([df,subd],axis=1)
    df.drop(['verification_status','application_type','initial_list_status','purpose'],axis=1,inplace=True)
    df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
    sube = pd.get_dummies(df['home_ownership'],drop_first=True)
    df = pd.concat([df,sube],axis=1)
    df['zip_code'] = df['address'].apply(lambda x: int(x.split()[-1]))
    df.drop(['home_ownership','address'],axis=1,inplace=True)
    subf = pd.get_dummies(df['zip_code'],drop_first=True)
    df = pd.concat([df,subf],axis=1)
    df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x: int(x.split('-')[1]))
    df.drop(['zip_code', 'issue_d', 'earliest_cr_line', 'loan_status'],axis=1,inplace=True)
    return df

def scale(df):
    X = df.drop('loan_repaid',axis=1).values
    y = df['loan_repaid'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def train(xtrain,ytrain):
    model = Sequential()
    model.add(Dense(78,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(39,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(19,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam')
    model.fit(x=xtrain, 
        y=ytrain, 
        epochs=2,
        batch_size=256,
        validation_data=(X_test, y_test), 
        )
    return model

def r_costumer(df):
    random.seed()
    random_ind = random.randint(0,len(df))

    new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
    new_status = df['loan_repaid'].iloc[random_ind]
    return new_customer,new_status

if __name__ == '__main__':
    df = opendata('lending_club_loan_two.csv')
    data_info = opendata('lending_club_info.csv')
    print(data_info)
    info(df)
    #visualization(df)
    df = drop(df)
    print(df)
    df = feature(df)
    X_train, X_test, y_train, y_test, scaler = scale(df)
    Model = train(X_train,y_train)
    pred = (Model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test,pred))
    print('\n')
    print(confusion_matrix(y_test,pred))
    new_customer,new_status = r_costumer(df)
    print(new_customer)
    print(new_status)
    nc = new_customer.values.reshape(1,78)
    print((Model.predict(scaler.transform(nc)) > 0.5).astype("int32"))
    Model.save('lending.h5')