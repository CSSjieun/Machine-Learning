# Machine Learning Method

## Chapter 7. Ensemble Learning and Random Forest
Gathering predictions from a set of predictors (classification or regression models) can produce better predictions than the best single model. 
Since a series of predictors is called an ensemble, this is called ensemble learning, and the ensemble learning algorithm is called an ensemble method.

An example of an ensemble method is to train a series of decision tree classifiers by randomly creating different subsets from the training set.
```python
from sklearn.datasets import make_moons
```
