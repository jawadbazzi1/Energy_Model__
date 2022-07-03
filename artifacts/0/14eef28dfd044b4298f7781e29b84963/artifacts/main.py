import mlflow
import pandas as pd
import numpy as np
import pickle
import os.path
from urllib.parse import urlparse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # run this command in the terminal before running
    # mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # read data from csv file
    energy_efficiency_data = pd.read_csv("energy_efficiency_data.csv")

    # check for data change
    noChange = True
    previous_data = pd.read_csv("previous.csv")

    if(len(energy_efficiency_data.index) != len(previous_data.index)):
        noChange = False
    else:
        noChange=energy_efficiency_data.equals(previous_data)


    # creating the attributes for the regression model
    X = energy_efficiency_data.drop(columns=['Heating_Load','Cooling_Load'])
    Y = energy_efficiency_data[['Heating_Load','Cooling_Load']]

    # check whether the model is trained: if it does load it, if not train it
    if os.path.exists("trained_model.pkl") and noChange:
        print("Loading Trained Model")
        model = pickle.load(open("trained_model.pkl","rb"))
    else:
        print("Creating and training a new model:")
        print("Model Traning")
        print("***********************************")

        #training will happen if no prevouis training is done
        best_accuracy = 0
        for i in range(100):

            #create test and train data sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

            # create the model
            model = LinearRegression()

            # train the model
            model.fit(X_train, Y_train)

            # calculate accuracy of the model
            accuracy = model.score(X_test,Y_test)

            # save model if and only accuracy is better than the previous one
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print("Accuracy",accuracy)

                # once the model is trained, save it to this file
                with open("trained_model.pkl","wb") as file:
                    pickle.dump(model,file)

        print("***********************************")
        print("Model Training Completed")

        # update previous data
        previous_data = energy_efficiency_data
        previous_data.to_csv("previous.csv", index=False)

        # load the model with best accuracy
        model = pickle.load(open("trained_model.pkl","rb"))


    # get accuracy score
    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    with mlflow.start_run():
        predictions = model.predict(X)
        (rmse, mae, r2) = eval_metrics(Y, predictions)

        print("  RMSE: %s" % rmse)
        print("  MAE:  %s" % mae)
        print("  R2:   %s" % r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # register model in mlflow
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="Energy Consumtion Model")
        else:
            mlflow.sklearn.log_model(model, "model")

        client = mlflow.tracking.MlflowClient()
        data = client.get_run(mlflow.active_run().info.run_id).data

        mlflow.log_artifact("energy_efficiency_data.csv")
        mlflow.log_artifact("main.py")
        mlflow.log_artifact("results.csv")
