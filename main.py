# Run different models on the dataset
import mlr
import decisionTree
import randomForest
import os
import time
import sys


def main():
    # run all models and compare them
    os.system('cls')
    print("Running models...")
    start = time.time()

    mlr_model, mlr_mse, _, _,_ = mlr.mlr()
    dt_model, dt_mse, dt_feature_importance= decisionTree.decision_tree()
    rf_model, rf_mse, rf_feature_importance,_,_ = randomForest.random_forrest()
    
    # Compare the models
    print("Mean Squared Error Comparison:")
    print(f"MLR: {mlr_mse}")
    print(f"Decision Tree: {dt_mse}")
    print(f"Random Forest: {rf_mse}")
    
    # check which model has the lowest MSE
    best_model = min([mlr_mse, dt_mse, rf_mse], key=lambda x: x)
    print(f"The best model is {best_model} with a MSE of {best_model}")
    end = time.time()
    print(f"Time taken to run all models: {end-start} seconds")

if __name__ == '__main__':
    sys.stdout = open("output.txt", "w")
    main()
    sys.stdout.close()
    sys.stdout = sys.__stdout__
