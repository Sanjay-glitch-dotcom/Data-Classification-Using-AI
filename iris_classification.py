import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
def main():
    print("Loading the Iris dataset")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    print("\n Dataset Information ")
    print("The Iris dataset contains measurements of 150 iris flowers from three different species.")
    print("Features (Input):")
    for feature in iris.feature_names:
        print(f" {feature}")
    print("\nTarget Labels (Output - the species to predict):")
    for i, target_name in enumerate(iris.target_names):
        print(f" Class {i}: {target_name}")
    print("\n Data Exploration ")
    print(f"Dataset Shape (rows, columns): {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\n Splitting Dataset ")
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Training data shape (features): {X_train.shape}")
    print(f"Testing data shape (features): {X_test.shape}")
    print("\n Training Model ")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    print("Model training complete. (Algorithm: Decision Tree Classifier)")
    print("\n Making Predictions ")
    y_pred = clf.predict(X_test)
    print("\n Model Evaluation ")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print("(Rows represent actual classes, columns represent predicted classes)")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("\n Sample Predictions vs Actual ")
    for i in range(5):
        actual_class = iris.target_names[y_test.iloc[i]]
        predicted_class = iris.target_names[y_pred[i]]
        print(f"Sample {i+1}: Predicted = {predicted_class}, Actual = {actual_class}")
if __name__ == "__main__":
    main()