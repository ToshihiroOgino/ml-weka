# Titanic

データセット: Titanic - Machine Learning from Disaster
URL: https://www.kaggle.com/competitions/titanic/overview

## 前処理

- Nameに含まれていたダブルクオートを削除した
- PassengerId, Name, Ticketの削除
- "Sex", "Embarked", "Cabin"をOne-Hot化。Cabinは先頭のアルファベットを利用(例: C85 -> C)
