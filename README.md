# ML-Weka

## Report

[report.md](report.md)

## Datasets

### Titanic: Machine Learning from Disaster

URL: https://www.kaggle.com/competitions/titanic/overview

#### 前処理

- Nameに含まれていたダブルクオートを削除した
- PassengerId, Name, Ticketの削除
- "Sex", "Embarked", "Cabin"をOne-Hot化。Cabinは先頭のアルファベットを利用(例: C85 -> C)


### Iris Species

URL: https://www.kaggle.com/datasets/uciml/iris

#### 前処理

- Idの削除

### House Prices - Advanced Regression Techniques

URL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

#### 前処理

- Id列を削除
- カテゴリ変数の数値化:
  - 品質と状態に関する変数（ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC）を順序尺度（Ex=5, Gd=4, TA=3, Fa=2, Po=1, NA=0）に変換
  - その他のカテゴリ変数を順序尺度に変換:
    - LotShape: Reg=3, IR1=2, IR2=1, IR3=0
    - LandSlope: Gtl=2, Mod=1, Sev=0
    - LandContour: Lvl=3, Bnk=2, HLS=1, Low=0
    - Utilities: ELO=3, NoSewr=2, NoSeWa=1, AllPub=0
    - PavedDrive: Y=2, P=1, N=0
    - その他（Electrical, Functional, GarageType, GarageFinish, Fence）も同様に順序尺度化
- 残りのカテゴリカル変数をOne-Hot化
