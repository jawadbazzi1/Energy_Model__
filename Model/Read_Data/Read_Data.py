import pandas as pd

import mlflow

def read_data():
    # run this command in the terminal before running
    # mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # read data from csv file
    energy_efficiency_data = pd.read_csv(r"C:\Users\reine\PycharmProjects\Energy_Model_\Model\Read_Data\energy_efficiency_data.csv")

    # check for data change
    noChange = True
    previous_data = pd.read_csv(r"C:\Users\reine\PycharmProjects\Energy_Model_\Model\Read_Data\previous.csv")

    if (len(energy_efficiency_data.index) != len(previous_data.index)):
        noChange = False
    else:
        noChange = energy_efficiency_data.equals(previous_data)

    return energy_efficiency_data,noChange