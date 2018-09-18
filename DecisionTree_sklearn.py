# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:30:06 2018

@author: DELL
"""

import numpy as np
import pandas as pd
from sklearn import tree

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus

input_file = "PastHires.csv"
df = pd.read_csv(input_file, header = 0)

d = {'Y' : 1, 'N' : 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS' : 0, 'MS' :1, 'PhD' : 2}
df['Level of Education'] = df['Level of Education'].map(d)
print(df.head())
#Notice data now in form of numbers

features = list(df.columns[:6])
y = df["Hired"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

dot_data = StringIO()
tree.export_graphviz(clf, out_file = dot_data, feature_names = features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

print("Creating graph..")
graph.write_png('abc.png')
