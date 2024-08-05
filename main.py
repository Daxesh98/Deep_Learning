from src.data.make_dataset import load_data
from src.features.build_features import dummy
from src.models.train_model import Deep_Learn, Model_MN, activation_function
from src.models.predict_model import evaluate_model
from src.visualization.visualize import semi_log, loss_curve


if __name__=="__main__":
    #Load Data
    data_path = "data\\raw\\employee_attrition.csv"
    df =load_data(data_path)
    
    # Preprocess Data
    Y, X, X_scaled = dummy(df)
    
    # Train and Evaluate Deep_Learn model
    evaluation = Deep_Learn(Y, X, X_scaled)
    print("Deep_Learn evaluation:", evaluation)
    
    # Train and Evaluate Model_MN model
    history = Model_MN(Y, X, X_scaled)
    semi_log(history)
    
    # Train and Evaluate model with activation_function
    y_test, pred, history1 = activation_function(Y, X, X_scaled)
    accuracy = evaluate_model(y_test, pred)
    print("Activation Function Model accuracy:", accuracy)
    
    # Visualize loss curve
    loss_curve(history1)