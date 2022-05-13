import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Iris
def split_iris_data(df):
    train, test = train_test_split(df, train_size=.8, random_state=123, stratify=df.species)
    
    train, validate = train_test_split(train, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train.species)
    return train, validate, test

def prep_iris(df):
    df = df.drop_duplicates()
    columns_to_drop = ['species_id','measurement_id']
    df = df.drop(columns = columns_to_drop)
    df = df.rename(columns={'species_name': 'species'})
    dummy_df = pd.get_dummies(df[['species']], drop_first = True)
    df = pd.concat([df, dummy_df], axis=1)

    train, validate, test = split_iris_data(df)
    
    return train, validate, test

#-------------------------------------------------------------------------

#Titanic
def split_titanic_data(df):
    train, test = train_test_split(df, train_size=.8, random_state=123, stratify=df.survived)
    
    train, validate = train_test_split(train, 
                                       train_size=.7, 
                                       random_state=123, 
                                       stratify=train.survived)
    return train, validate, test

def prep_titanic(df):
    df = df.drop_duplicates()
    columns_to_drop = ['passenger_id','deck','class','embarked']
    df = df.drop(columns = columns_to_drop).rename(columns={'sibsp': 'Siblings/Spouses','parch':'Parents/Children'})
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)

    train, validate, test = split_titanic_data(df)

    return train, validate, test

#-------------------------------------------------------------------------

#Telco
def split_telco_data(df):
    train, test = train_test_split(df, train_size=.8, random_state=123, stratify=df.churn)
    
    train, validate = train_test_split(train, 
                                       train_size=.7, 
                                       random_state=123, 
                                       stratify=train.churn)
    return train, validate, test

def prep_telco(df):
    df = df.drop_duplicates()
    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    df['total_charges'] = df.total_charges.astype(float)
    
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    columns_to_drop = ['monthly_charges.1','customer_id', 'total_charges.1','paperless_billing.1','payment_type_id.1','internet_service_type_id','contract_type_id','payment_type_id']
    df = df.drop(columns = columns_to_drop).rename(columns={'sibsp': 'Siblings/Spouses','parch':'Parents/Children'})
    
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type']], dummy_na=False, \
                              drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    train, validate, test = split_telco_data(df)

    return train, validate, test
