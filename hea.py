import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import openpyxl
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import genfromtxt
import joblib


def main():
    df = pd.read_csv('dt1v4.csv', sep=';')
    df = df.astype(float)
    X = df.drop(['sortie_risque', 'classe_risque_actuelle'], axis=1)
    y = df.sortie_risque
    # Scaling our columns
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    # Diviser les données (70% Apprentissage et 30% Test)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model_rf = RandomForestClassifier(n_estimators=650, criterion="gini", random_state=42, n_jobs=-1)
    model_rf.fit(x_train, y_train)

    model_bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=11, n_jobs=-1,
                                 n_estimators=100, random_state=42)
    model_bg.fit(x_train, y_train)

    model_lr = LogisticRegression(random_state=42)
    model_lr.fit(x_train, y_train)

    model_xg = GradientBoostingClassifier(random_state=42)
    model_xg.fit(x_train, y_train)

    model_adab = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100, learning_rate=1, random_state=42)
    model_adab.fit(x_train, y_train)

    model_tree = DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=7)
    model_tree.fit(x_train, y_train)

    model_mpl = MLPClassifier(random_state=42)
    model_mpl.fit(x_train, y_train)

    votingClassifier = VotingClassifier(
        estimators=[('rf', model_rf), ('dt', model_tree), ('bg', model_bg), ('adab', model_adab), ('lr', model_lr),
                    ('mpl', model_mpl), ('xg', model_xg)], voting='soft', n_jobs=-1)

    votingClassifier.fit(x_train, y_train)

    print('Accuracy: ', votingClassifier.score(x_test, y_test))
    # enregistirer le model comme a pickle dans un dossier
    joblib.dump(VotingClassifier, 'mo.pkl')

    with open('modelv2.pkl', 'wb') as f:
        pickle.dump(votingClassifier, f)


if __name__ == '__main__':
    main()
