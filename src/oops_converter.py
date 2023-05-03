import pickle 
import os
import numpy as np
from generate_processed_data import PCA_inverse_transform

THISFILE_PATH = os.path.abspath(__file__)
DATAOUT_PATH = os.path.join(
    os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), "dataout"
)

def correct_incorrect_model_output(subject):

    with open(f"{DATAOUT_PATH}/predictions/EffecientNet/{subject}/y_test_effecientnet_{model_version}.pickle", "rb") as f:
        y_test = pickle.load(f)   
    with open(f"{DATAOUT_PATH}/predictions/EffecientNet/{subject}/y_pred_effecientnet_{model_version}.pickle", "rb") as f:
        y_pred = pickle.load(f)   

    print("y_pred shape: ", y_pred.shape)
    print("y_test shape: ", y_pred.shape)
    print("")

    y_pred = PCA_inverse_transform(y_pred, subject) # Inverts into vector
    y_test = PCA_inverse_transform(y_test, subject) # Inverts into vector

    with open(f"{DATAOUT_PATH}/predictions/EffecientNet/{subject}/y_test_effecientnet_{model_version}_new.pickle", "wb") as f:
        pickle.dump(y_test, f)   
    with open(f"{DATAOUT_PATH}/predictions/EffecientNet/{subject}/y_pred_effecientnet_{model_version}_new.pickle", "wb") as f:
        pickle.dump(y_pred, f)   

    print("\nAFTER INVERTING PCA\n")
    print("y_pred shape: ", y_pred.shape)
    print("y_test shape: ", y_pred.shape)
    print("")

if __name__ == "__main__":
    subject = 'subj01'
    model_version = "2"

    
    correct_incorrect_model_output(subject)
