from Model.Model_Evaluation.Model_Evaluation import model_evaluation
from Model.Model_Training.Model_Training import model_training
from Model.Read_Data.Read_Data import read_data

energy_efficiency_data,noChange=read_data()
model,X,Y = model_training(energy_efficiency_data,noChange)
model_evaluation(model,X,Y)

#pipeline = Pipeline[("read data",read_data()),
#                    ("train model",model_training(energy_efficiency_data,noChange)),
#                   ("evaluate model"),model_evaluation(model)]
