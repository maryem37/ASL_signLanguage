import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('E:/Hackathons/LOC 6.0/sign-language-detector-python-master/data.pickle', 'rb'))

data = []
labels = []

max_length = max(len(np.array(item).flatten()) for item in data_dict['data'])

for item in data_dict['data']:
    flattened_item = np.array(item).flatten()
    padded_item = np.pad(flattened_item, (0, max_length - len(flattened_item)))
    data.append(padded_item)


labels = np.array(data_dict['labels'])
data = np.array(data)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
