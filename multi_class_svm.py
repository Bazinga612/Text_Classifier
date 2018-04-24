from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def run_classifier(X_train,X_test,y_train,y_test):
    #building the Classifier
    classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC(loss='squared_hinge', dual=False, tol=1e-3))])
    print("--------------------------classifying-------------------------------")
    #training the classifier
    classifier.fit(X_train, y_train)
    #predicting the values
    predicted = classifier.predict(X_test)
    #calculating the accuracy
    accuracy = metrics.accuracy_score(y_test, predicted)
    print("accuracy is ",accuracy)
