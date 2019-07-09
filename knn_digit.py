from sklearn.datasets import load_digits

dataset = load_digits()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)


from sklearn.model_selection import cross_val_score

scores = []
for i in range(1,21):
    clf = KNeighborsClassifier(n_neighbors=i)
    scores.append(cross_val_score(clf,X,y,cv=4).mean())

import matplotlib.pyplot as plt
plt.plot(range(1,21),scores)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Finding optimal K")

model = KNeighborsClassifier(n_neighbors=1)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)


