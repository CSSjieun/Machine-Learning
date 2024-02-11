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

This code trains an ensemble of 500 decision tree classifiers.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, max_samples = 100, n_jobs = -1, random_state = 42)

bag_clf.fit(X_train, y_train)
```

Bagging has a slightly higher bias than pasting because it adds diversity to the subset that each predictor learns, but adding diversity means that the variance of the ensemble is reduced by reducing the correlation between predictors.

### 7.2.2. OOB (Out-of-Bag) Score

Using bagging means that, on average, only about 63% of the training samples are sampled for each predictor. <br/>
The left 37% is called OOB (out-of-bag) sample.

```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, oob_score = True, n_jobs = -1, random_state = 42)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_
```
0.896

Based on the OOB evaluation results, it appears that this BaggingClassifier will achieve approximately 89.6% accuracy on the test set.

Let's check it.
```python
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```
0.9120000000

We got about 92% accuracy on the test set. The OOB score was slightly pessimistic, falling by more than 2%.

```python
bag_clf.oob_decision_function_[:3]
```
array([[0.32352941, 0.67647059],
       [0.3375    , 0.6625    ],
       [1.        , 0.        ]])
       
→ The OOB score estimates a 67.64% probability that the first training sample will belong to the positive class and a 32.4% probability that it will belong to the negative class.

### 7.4 Random Forest

Below is code to train a random forest classifier with 500 trees (with up to 16 leaf nodes) on all available CPU cores.

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, n_jobs = -1, random_state = 42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

The random forest algorithm injects more randomness by finding the best feature among randomly selected feature candidates when splitting nodes in a tree, instead of finding the best feature among all features.

**BaggingClassifier**

```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(max_features = "sqrt", max_leaf_nodes = 16), n_estimators = 500, n_jobs = -1, random_state = 42)
```

Such extremely random tree's random forest is called extremely randomized tree ensemble (extra-tree).

To make extra tree, we can use the code of ExtraTreesClassifier from scikit-learn.

### 7.4.2 Feature Importance

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame = True)
rnd_clf = RandomForestClassifier(n_estimators = 500, random_state = 42)
rnd_clf.fit(iris.data, iris.target)
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
  print(round(score, 2), name)
```

0.11 | sepal length (cm)

0.02 | sepal width (cm)

0.44 | petal length (cm)

0.42 | petal width (cm)

