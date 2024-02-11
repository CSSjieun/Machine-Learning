# Machine Learning Method

## Chapter 7. Ensemble Learning and Random Forest
Gathering predictions from a set of predictors (classification or regression models) can produce better predictions than the best single model. <br/>
Since a series of predictors is called an ensemble, this is called ensemble learning, and the ensemble learning algorithm is called an ensemble method.

An example of an ensemble method is to train a series of decision tree classifiers by randomly creating different subsets from the training set. <br/>
Predictions from individual trees are pooled, and the most selected class is used as the prediction for the ensemble. <br/>
An ensemble of decision trees is called a random forest.

### Voting Classifiers
Suppose we have trained several classifiers (logistic regression classifier, SVM classifier, random forest classifier, k-nearest neighbor classifier, etc.) with 80% accuracy. <br/>
A way to create better classifiers is to aggregate the predictions of each classifier. <br/>
The class that receives the most votes becomes the ensemble prediction. <br/>
This method is **hard voting classifiers**.

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples = 500, noise = 0.30, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

voting_clf = VotingClassifier(
  estimators = [
    ("lr", LogisticRegression(random_state=42)),
    ("rf", RandomForestClassifier(random_state=42)),
    ("svc", SVC(random_state=42))
]
)
voting_clf.fit(X_train, y_train)
```

First, we will check the accuracy of each trained classifiers from the test set.

```python
for name, clf in voting_clf.named_estimators_.items():
  print(name, "=", clf.score(X_test, y_test))
```

lr =  0.864 <br/>
rf = 0.896 <br/>
svc = 0.896

predict() method operate **hard voting**.

```python
voting_clf.predict(X_test[:1])
[clf.predict(X_test[:1]) for clf in voting_clf.estimators_]

voting_clf.score(X_test, y_test)
```

0.912

→ As expected, the voting classifier performs slightly better than the other individual classifiers.

Using **soft voting**
If all classifiers can predict the probability of a class (i.e., if there is a predict_proba() method), the predictions of individual classifiers can be averaged to predict the class with the highest probability.

```python
voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)
```
0.92
→ Using soft voting, we achieved 92% accuracy.



