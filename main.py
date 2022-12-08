from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

def classify(train_data, test_data):
    # Preprocess the train and test data to extract the review text and ratings
    train_text, train_labels = preprocess_data(train_data)
    test_text, test_labels = preprocess_data(test_data)
    
    # Convert the text data into a matrix of token counts
    vectorizer = CountVectorizer()
    train_vectors = vectorizer.fit_transform(train_text)
    test_vectors = vectorizer.transform(test_text)
    
    # Train an SVM model on the training data
    svm = SVC()
    svm.fit(train_vectors, train_labels)
    
    # Use the trained model to make predictions on the test data
    predictions = svm.predict(test_vectors)
    
    # Compute the F1 score and overall accuracy for each class
    f1_scores = {}
    for label in set(test_labels):
        f1 = f1_score(test_labels, predictions, pos_label=label)
        f1_scores[label] = f1
    
    overall_accuracy = accuracy_score(test_labels, predictions)
    
    # Return the F1 scores and overall accuracy as a dictionary
    return {'F1 scores': f1_scores, 'Overall accuracy': overall_accuracy}
    
    
