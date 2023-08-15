import pandas as pd
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor

token = subprocess.run('gcloud auth print-identity-token', shell=True, capture_output=True, text=True).stdout.strip()
headers = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}

"""
begin date: '2017-04-08 23:36:00'
end_date: '2017-07-19 23:59:00'

Solar event was on 27 Sep 2017
"""

def get_date_list(begin_date, end_date):
    start_date = pd.to_datetime(begin_date)
    end_date = pd.to_datetime(end_date)
    date_list = pd.date_range(start=start_date, end=end_date, freq='12min').tolist()
    return date_list

def eve_endpoint(data):
    result = requests.post(
        url="https://us-central1-us-fdl-x.cloudfunctions.net/virtual-eve",
        headers=headers,
        json=data,
    )
    return result

date_list = get_date_list(begin_date='2017-04-08 23:36:00', end_date='2017-07-19 23:59:00')

futures = []
with ThreadPoolExecutor(max_workers=300) as executor:
    for inference_date in date_list:
        timestamp = inference_date.strftime('%Y-%m-%dT%H:%M:%S')
        print(f"Submitting: {timestamp} to thread pool")

        message = {"time": timestamp}
        futures.append(executor.submit(eve_endpoint, message))

    results = [future.result() for future in futures]

# for result in results:
#     print(result.text)

print(f"Submitted {len(date_list)} requests to EVE endpoint.")
print(f"Received {len(results)} responses from EVE endpoint.")