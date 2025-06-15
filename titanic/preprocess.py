import pandas as pd
import os

# データセットのパスを設定
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'titanic')

# CSVファイルを読み込む
train_df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
gender_submission_df = pd.read_csv(os.path.join(DATASET_DIR, 'gender_submission.csv'))

# データの確認
print("Training Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print("Gender Submission Shape:", gender_submission_df.shape)

# データの先頭を表示
print("\nTraining Data Head:")
print(train_df.head())
