import pandas as pd
df = pd.read_csv(r"C:\IP\greenmlops\airflow\include\data\processed\ett\train.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(2))



import pandas as pd
df_val  = pd.read_csv(r"C:\IP\greenmlops\airflow\include\data\processed\ett\val.csv")
df_test = pd.read_csv(r"C:\IP\greenmlops\airflow\include\data\processed\ett\test.csv")
print(df_val.shape, df_test.shape)