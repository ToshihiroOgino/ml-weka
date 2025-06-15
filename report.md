# Wekaを用いた機械学習の実践

2025年6月15日

AL21066 荻野 寿大

## 学習手法の概要

本レポートでは、Wekaに実装されている学習手法の中から、**J48**、**SMO**、**Random Forest**の3つを選択した。それぞれの特徴は以下の通りである。

### J48

J48は、C4.5アルゴリズムをJavaで実装した**決定木**学習アルゴリズムである。与えられたデータから、木構造の分類モデルを生成する。各分岐では、情報利得率が最大となる属性を選択してデータを分割していく。このプロセスを再帰的に繰り返すことで、最終的な決定木が構築される。生成されたモデルは「if-then」形式のルールとして解釈できるため、結果の**可読性が高く**、どのような特徴量が分類に寄与しているかを理解しやすい点が大きな特徴である。

### SMO

SMO (Sequential Minimal Optimization) は、**サポートベクターマシン (SVM)** を効率的に学習させるためのアルゴリズムである。SVMは、データ点を2つのクラスに分類する際に、その境界（超平面）と最も近いデータ点（サポートベクター）との距離（マージン）が最大になるように学習する。SMOは、この複雑な最適化問題を、一度に2つのデータ点のみを対象とする小さな部分問題に分割し、それを逐次的に解くことで高速な計算を実現する。これにより、大規模なデータセットに対しても効率的にSVMを適用することが可能となる。

### Random Forest

Random Forestは、複数の決定木を組み合わせることで予測性能を高める**アンサンブル学習**の一種である。学習データからランダムにデータをサンプリングし、それぞれのデータセットで決定木を構築する。さらに、各決定木の分岐を作成する際には、全ての特徴量からではなく、ランダムに選ばれた一部の特徴量の中から最適なものを選択する。これにより、多様性に富んだ複数の決定木が生成される。最終的な予測は、分類問題では多数決、回帰問題では平均値によって決定される。個々の決定木が過学習を起こしていても、その平均を取ることで汎化性能の高い安定したモデルとなる。

---

## 学習結果と考察

性質の異なる3つのデータセットに対し、前述の学習手法を適用し、その結果を比較・考察する。
対象のデータセットは以下の3つである。

- Titanic: Machine Learning from Disaster
    https://www.kaggle.com/competitions/titanic/overview
- Iris Species
    https://www.kaggle.com/datasets/uciml/iris
- House Prices: Advanced Regression Techniques
    https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

### データセットの前処理

各データセットに対して行った前処理の例を以下に示す。詳細な前処理に関するコードはGitHubリポジトリにて公開している (https://github.com/ToshihiroOgino/ml-weka)。

- ID列や人名など、学習に不要な列の削除
- 品質と状態に関する変数を順序尺度に変換 (例: ExterQual について Ex=5, Gd=4, TA=3, Fa=2, Po=1, NA=0)
- カテゴリ変数のOne-Hot化
- 学習手法に応じた正解ラベルの変換 (Survivedの0,1をDied, Survivedに変換)
- TitanicのCabin列をOne-Hot化 (例: C123 -> Cabin_C: True)


### データセットと結果

各データセットに適用した手法と得られた主要な評価指標は以下の通りである。

いずれのデータセットも、100~3000件程度のデータしか提供されていなかったため、Test optionsは全てのデータセットで10-fold cross-validationを使用した。

| データセット     | 手法          | 主要評価指標 | 結果   |
| :--------------- | :------------ | :----------- | :----- |
| **Titanic**      | J48           | 正解率       | 86.25% |
| **Iris Species** | SMO           | 正解率       | 96.00% |
| **House Prices** | Random Forest | 相関係数     | 0.6939 |

各学習におけるWekaのSummaryなどに関する出力は以下の通りである。

#### Titanic: Machine Learning from Disaster

```plaintext
J48 pruned tree
------------------

Sex_male = True
|   Cabin_E = False
|   |   Age <= 9
|   |   |   SibSp <= 2
|   |   |   |   Fare <= 15.55: Died (17.75/3.51)
|   |   |   |   Fare > 15.55: Survived (18.77/4.25)
|   |   |   SibSp > 2: Died (16.32/1.0)
|   |   Age > 9: Died (768.16/79.97)
|   Cabin_E = True
|   |   Pclass <= 2
|   |   |   Parch <= 0
|   |   |   |   Age <= 44: Survived (6.4/0.4)
|   |   |   |   Age > 44: Died (9.6/2.0)
|   |   |   Parch > 0: Died (4.0)
|   |   Pclass > 2: Survived (2.0)
Sex_male = False
|   Pclass <= 2: Survived (250.0/9.0)
|   Pclass > 2
|   |   Fare <= 23.25: Survived (183.0/48.0)
|   |   Fare > 23.25
|   |   |   Parch <= 0: Survived (2.0)
|   |   |   Parch > 0: Died (31.0/7.0)

Correctly Classified Instances        1129               86.249  %
Incorrectly Classified Instances       180               13.751  %
Kappa statistic                          0.7024
Mean absolute error                      0.2159
Root mean squared error                  0.3362
Relative absolute error                 45.9384 %
Root relative squared error             69.3491 %
Total Number of Instances             1309
```

#### Iris Species

```plaintext
Correctly Classified Instances         144               96      %
Incorrectly Classified Instances         6                4      %
Kappa statistic                          0.94
Mean absolute error                      0.2311
Root mean squared error                  0.288
Relative absolute error                 52      %
Root relative squared error             61.101  %
Total Number of Instances              150
```

#### House Prices: Advanced Regression Techniques

```plaintext
Correlation coefficient                  0.6939
Mean absolute error                  28913.6814
Root mean squared error              41486.2094
Relative absolute error                 83.4815 %
Root relative squared error             72.3054 %
Total Number of Instances             2919
```

### 考察

#### Titanic (J48)

タイタニック号の乗客の生存予測という**分類問題**に対して、解釈性の高いJ48を適用した。正解率は**86.25%**と良好な結果を示した。
生成された決定木を見ると、最初の分岐が`Sex_male`(性別が男性か)となっており、生存予測において性別が最も重要な要因であることがわかる。これは、歴史的事実である「女性と子供が優先的に救助された」という状況をモデルが正しく捉えていることを示唆している。その他にも、客室の等級や年齢、運賃などが生存を左右する要因として抽出されており、J48の**解釈性の高さ**が、単なる予測精度だけでなく、データ背景の理解にも貢献していることが確認できた。Cabinのように、文字列で表現されるデータを適切に前処理を行い学習に利用したことが、精度向上に寄与したと考えられる。

#### Iris Species (SMO)

アヤメの品種分類という古典的な**分類問題**にSMOを適用した結果、正解率は**96.00%**という非常に高い値になった。Iris Speciesデータセットは特徴量数が少なく、クラス間の分離が比較的明確であるため、マージン最大化によって最適な分離境界を探索するSVM(およびその学習アルゴリズムであるSMO)が効果的に機能したと考えられる。この結果は、比較的単純な構造を持つデータに対して、SMOが高い分類性能を発揮する好例といえる。

#### House Prices (Random Forest)

住宅価格を予測する**回帰問題**に対し、Random Forestを適用した。このデータセットは特徴量が多く、欠損値や多様なデータ型(数値、カテゴリカル)を含む複雑なものである。適切な前処理(カテゴリ変数の順序尺度化やOne-Hotエンコーディング)を施した上で学習させた結果、相関係数は**0.6939**となった。
この数値は、モデルがある程度の予測能力を持つことを示している。多数の決定木を組み合わせるRandom Forestは、特徴量が多く複雑なデータに対しても過学習を抑制し、**安定した予測モデルを構築できる**という強みが発揮された結果である。Mean Absolute Errorが約28,913ドルであったことから、平均してこの程度の誤差で価格を予測できるモデルが構築されたことがわかる。このデータセットは、One-Hot化を行ったことで説明変数の数が200程度に増加しており、精度の改善を図るためには、さらに特徴量選択や次元削減などの手法を検討する余地がある。

#### 総合考察

今回の実践を通じて、データセットの特性(問題の種類、データの複雑さ、特徴量の数など)に応じて適切な学習手法を選択することの重要性が確認できた。
- **解釈性**が求められる場面では**J48(決定木)**
- **高い分類精度**が期待できる場面では**SMO (SVM)**
- **複雑でノイズの多いデータ**を扱う場面では**Random Forest**
といった使い分けが有効である。また、特にHouse Pricesの例では、ドメイン知識に基づいた前処理(品質に関するカテゴリ変数の順序尺度化など)がモデルの性能に大きく影響することも示唆された。機械学習においては、アルゴリズムの選択だけでなく、データそのものへの深い理解と適切な前処理が不可欠であると結論付けられる。

## Wekaを利用した感想

Wekaで提供されている前処理ツールの利用方法が分かりづらく、学習コストが高いと感じたため、今回のレポート作成の際にはPythonのPandasを用いて前処理を行った。

WekaのGUIは直感的で使いやすいが、特に複雑な前処理を行う場合には、Pythonなどのプログラミング言語を用いた方が、操作が明確で効率的だと感じた。
また、近年ではKerasやTensorFlowなどのディープラーニングフレームワークが普及し、関連情報も得やすいため、WekaのようなGUIベースの機械学習ツールを利用する利点は相対的に少なくなってきていると感じた。
