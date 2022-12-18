import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from time import time
import re

start = time()


def classify(train_file, test_file):
    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')

    # Read json data files
    with open(train_file, 'r') as f:
        train_data = [json.loads(line) for line in f]
    with open(test_file, 'r') as f:
        test_data = [json.loads(line) for line in f]

    # Get rid of reviews with no text or summary
    train_reviews = list(filter(lambda review: 'reviewText' in review and 'summary' in review, train_data))
    test_reviews = list(filter(lambda review: 'reviewText' in review and 'summary' in review, test_data))

    # Remove punctuation
    punctuation_pattern = r'[^a-zA-Z\s]'

    train_review_texts = [re.sub(punctuation_pattern, '', review['summary'] + " " + review['reviewText']) for review in train_reviews]
    train_review_ratings = [review['overall'] for review in train_reviews]
    test_review_texts = [re.sub(punctuation_pattern, '', review['summary'] + " " + review['reviewText']) for review in test_reviews]
    test_review_ratings = [review['overall'] for review in test_reviews]

    # Create feature vectors using TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000)

    x_train = vectorizer.fit_transform(train_review_texts)
    y_train = train_review_ratings
    x_test = vectorizer.transform(test_review_texts)
    y_test = test_review_ratings
    words = vectorizer.get_feature_names_out()

    # Extract the 15 features with the highest discriminative power
    selector = SelectKBest(k=15)
    selector.fit(x_train, y_train)
    best_features = selector.get_support()
    features = np.array(words)

    print(f'Select best 15 feature words: {", ".join(features[best_features])}')

    # Train a classifier on the train data
    clf = LogisticRegression(max_iter=20000)
    clf.fit(x_train, y_train)

    # Use the trained classifier to make predictions on the test data
    y_pred = clf.predict(x_test)

    # Evaluate the classifier's performance using the F1 metric and overall accuracy
    f1 = f1_score(y_pred, y_test, average=None)
    acc = accuracy_score(y_pred, y_test)

    # Print confusion matrix
    cm = confusion_matrix(y_pred, y_test)
    print(f'confusion matrix:\n{cm}')

    # Fill in the dictionary below with actual scores obtained on the test data
    return {f'class_{cls}_F1': score for cls, score in enumerate(f1, start=1)} | {'accuracy': acc}


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    for k, v in results.items():
        print(k, v, sep=' = ')

    print(f'total time: {time() - start:.2f}')
