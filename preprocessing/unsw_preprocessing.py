import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_unsw_data():

    # ==========================
    # Load dataset
    # ==========================

    train = pd.read_csv(
        "data/UNSW-NB15/UNSW_NB15_training-set.csv"
    )

    test = pd.read_csv(
        "data/UNSW-NB15/UNSW_NB15_testing-set.csv"
    )


    # ==========================
    # Separate labels
    # ==========================

    X_train = train.drop(
        "label",
        axis=1
    )

    y_train = train["label"]


    X_test = test.drop(
        "label",
        axis=1
    )

    y_test = test["label"]



    # ==========================
    # Remove unnecessary columns
    # ==========================

    remove_columns = [
        "id",
        "attack_cat"
    ]


    X_train = X_train.drop(
        columns=remove_columns,
        errors="ignore"
    )

    X_test = X_test.drop(
        columns=remove_columns,
        errors="ignore"
    )



    # ==========================
    # One-hot encoding
    # ==========================

    categorical_columns = [
        "proto",
        "service",
        "state"
    ]


    X_train = pd.get_dummies(
        X_train,
        columns=categorical_columns
    )


    X_test = pd.get_dummies(
        X_test,
        columns=categorical_columns
    )



    # ==========================
    # Align features
    # ==========================

    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0
    )



    # ==========================
    # Normalization
    # ==========================

    scaler = StandardScaler()


    X_train = scaler.fit_transform(
        X_train
    )


    X_test = scaler.transform(
        X_test
    )



    # ==========================
    # Split TESTING set
    # into validation and testing
    # ==========================

    X_val, X_test, y_val, y_test = train_test_split(

        X_test,

        y_test,

        test_size=0.5,

        random_state=42,

        stratify=y_test
    )



    # ==========================
    # Convert for RNN
    # ==========================

    X_train = X_train.reshape(
        X_train.shape[0],
        1,
        X_train.shape[1]
    )


    X_val = X_val.reshape(
        X_val.shape[0],
        1,
        X_val.shape[1]
    )


    X_test = X_test.reshape(
        X_test.shape[0],
        1,
        X_test.shape[1]
    )



    # ==========================
    # Save
    # ==========================

    np.save(
        "data/X_train.npy",
        X_train
    )

    np.save(
        "data/y_train.npy",
        y_train
    )


    np.save(
        "data/X_val.npy",
        X_val
    )

    np.save(
        "data/y_val.npy",
        y_val
    )


    np.save(
        "data/X_test.npy",
        X_test
    )

    np.save(
        "data/y_test.npy",
        y_test
    )



    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Testing data:", X_test.shape)

    print(
        "UNSW preprocessing completed"
    )



if __name__ == "__main__":
    load_unsw_data()
