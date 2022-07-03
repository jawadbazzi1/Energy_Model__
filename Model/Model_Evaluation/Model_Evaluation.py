import mlflow
from pathlib import Path
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def model_evaluation(model,X,Y):

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


        mlflow.log_artifact(r"C:\Users\rabih\PycharmProjects\Energy_MLFlow\Model\Model_Evaluation")
        mlflow.log_artifact(r"C:\Users\rabih\PycharmProjects\Energy_MLFlow\Model\Model_Training")
        mlflow.log_artifact(r"C:\Users\rabih\PycharmProjects\Energy_MLFlow\Model\Read_Data")
        mlflow.log_artifact(r"C:\Users\reine\PycharmProjects\Energy_Model_\results.csv")
