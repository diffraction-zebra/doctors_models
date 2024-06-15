import requests
from io import StringIO

import pandas as pd


######################-predict-##############################

start, end = '2024-01-29', '2024-02-18'

url = f'http://127.0.0.1:8000/models/predict'
r = requests.get(url, params={'start': start, 'end': end})

df = pd.read_json(StringIO(r.json()))
print(df.index)
print(df['МРТ'])

#######################-update-##########################
updates = df

url = f'http://127.0.0.1:8000/models/update'
requests.post(url, params={'data': df.to_json()})
