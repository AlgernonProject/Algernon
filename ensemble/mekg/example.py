from mekg import MEClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

rs = 2022
list_of_experts = [LogisticRegression(max_iter=2000, random_state=rs), 
                   MLPClassifier(hidden_layer_sizes=3,
                                 max_iter=2000,
                                 random_state=rs)]

"""
Recall that in a real scenario, you should optimize gaussian
threshold and covariance scalar factor accordingly to your
train data (after normalizing it).
"""
me = MEClassifier(base_estimators=list_of_experts,
                  n_estimators=2,
                  gaussian_threshold=0.5,
                  cov_factor=10,
                  random_state=rs)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=rs,
                                                    stratify=y)

me.fit(X_train, y_train)
print(me.score(X_test, y_test))

