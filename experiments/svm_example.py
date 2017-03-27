import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

y = [0, 1, 0, 1, 0, 1 ]


clf = svm.SVC(kernel='linear', C = 1.0)

clf.fit(X, y)

print(clf.predict([0.58, 0.76]))
print(clf.predict([10.58,10.76]))

