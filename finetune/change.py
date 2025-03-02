from datasets import Dataset
from datasets import load_dataset
import pandas as pd

# 将json文件转换为csv文件并保存
df = pd.read_json('huanhuan.json')
ds = Dataset.from_pandas(df)
ds.to_csv('huanhuan.csv')