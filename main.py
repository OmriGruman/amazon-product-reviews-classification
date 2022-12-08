def classify(train_data, test_data):
    # Create feature vectors using CountVectorizer or TfidfVectorizer
    vectorizer = CountVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform([review['reviewText'] for review in train_data])
    X_test = vectorizer.transform([review['reviewText'] for review in test_data])
    
    # Extract the 15 features with the highest discriminative power
    selector = SelectKBest(k=15)
    X_train = selector.fit_transform(X_train, [review['overall'] for review in train_data])
    X_test = selector.transform(X_test)
    
    # Train a classifier, such as a LinearSVC or a random forest classifier, on the train data
    clf = LinearSVC()
    clf.fit(X_train, [review['overall'] for review in train_data])
    
    # Use the trained classifier to make predictions on the test data
    y_pred = clf.predict(X_test)
    
    # Evaluate the classifier's performance using the F1 metric and overall accuracy
    f1 = f1_score(y_pred, [review['overall'] for review in test_data], average=None)
    acc = accuracy_score(y_pred, [review['overall'] for review in test_data])
    
    # Print the confusion matrix
    print(confusion_matrix(y_pred, [review['overall'] for review in test_data]))
    
    # Return the F1 metric and overall accuracy
    return {'f1': f1, 'accuracy': acc}
