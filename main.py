import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def classify(train_file, test_file):
    # todo: implement this function
    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')

    # todo: you can try working with various classifiers from sklearn:
    #  https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    #  https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #  please use the LogisticRegression classifier in the version you submit

    # Read json data files
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    # Create feature vectors using CountVectorizer or TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform([review['reviewText'] for review in train_data])
    X_test = vectorizer.transform([review['reviewText'] for review in test_data])

    # Extract the 15 features with the highest discriminative power
    selector = SelectKBest(k=15)
    X_train = selector.fit_transform(X_train, [review['overall'] for review in train_data])
    X_test = selector.transform(X_test)

    # Train a classifier, such as a LinearSVC or a random forest classifier, on the train data
    clf = LogisticRegression()
    clf.fit(X_train, [review['overall'] for review in train_data])

    # Use the trained classifier to make predictions on the test data
    y_pred = clf.predict(X_test)

    # Evaluate the classifier's performance using the F1 metric and overall accuracy
    f1 = f1_score(y_pred, [review['overall'] for review in test_data], average=None)
    acc = accuracy_score(y_pred, [review['overall'] for review in test_data])

    # Print the confusion matrix
    print(confusion_matrix(y_pred, [review['overall'] for review in test_data]))

    # todo: fill in the dictionary below with actual scores obtained on the test data
    test_results = {
        'class_1_F1': 0.0,
        'class_2_F1': 0.0,
        'class_3_F1': 0.0,
        'class_4_F1': 0.0,
        'class_5_F1': 0.0,
        'accuracy': 0.0
    }

    return test_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    for k, v in results.items():
        print(k, v)
