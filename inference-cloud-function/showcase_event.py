import pandas as pd

start_date = pd.to_datetime('2017-09-26')
end_date = pd.to_datetime('2017-09-28')

date_list = pd.date_range(start=start_date, end=end_date, freq='12S').tolist()

print(date_list)