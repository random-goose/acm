from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

newsgroup_train = fetch_20newsgroups(subset='train', shuffle=True)

X_train = newsgroup_train.data
y_train = newsgroup_train.target

newsgroup_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test = newsgroup_test.data
y_test = newsgroup_test.target

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

classifier = MultinomialNB()

classifier.fit(X_train_tfidf, y_train)

y_pred = classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred, target_names=newsgroup_train.target_names)
print("Classification Report:\n", report)