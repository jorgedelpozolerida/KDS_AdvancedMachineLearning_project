import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import pickle


def create_train_test_split(X_data, y_data, test_size=0.20, random_state=123):
    """
    Create train-test split
    """
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size,
                                                        random_state=random_state, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size,
                                                        random_state=random_state, shuffle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test 