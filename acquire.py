import pandas as pd
import os
from env import get_db_url


def get_titanic_data():
    filename = 'titanic_data.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        df = pd.read_sql(
            '''
            SELECT * FROM passengers
            ''',
            get_db_url('titanic_db')
        )

        df.to_file(filename)

        return df


def get_iris_data():
    filename = 'iris_data.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        df = pd.read_sql(
            '''
            SELECT * FROM species
            ''',
            get_db_url('iris_db')
        )

        df.to_file(filename)

        return df


def get_telco_data():
    filename = 'telco_churn_data.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        df = pd.read_sql(
            '''
            SELECT * FROM customer_contracts 
            JOIN contract_types USING(contract_type_id) 
            JOIN customer_payments USING(customer_id) 
            JOIN customers USING(customer_id) 
            JOIN internet_service_types USING(internet_service_type_id);
            ''',
            get_db_url('telco_churn')
        )

        df.to_file(filename)

        return df

