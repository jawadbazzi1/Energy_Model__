import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def model_training(data,checkTraining):

    # creating the attributes for the regression model
    X = data.drop(columns=['Heating_Load','Cooling_Load'])
    Y = data[['Heating_Load','Cooling_Load']]

    # check whether the model is trained: if it does load it, if not train it
    if os.path.exists("trained_model.pkl") and checkTraining:
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
        previous_data = data
        previous_data.to_csv("previous.csv", index=False)

        # load the model with best accuracy
        model = pickle.load(open("trained_model.pkl","rb"))

    return model,X,Y
