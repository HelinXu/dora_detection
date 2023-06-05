
# import requests
# import json

# url = "https://apps.beam.cloud/0oapt"
# payload = {"text": "Helin says hello!"}
# headers = {
#   "Accept": "*/*",
#   "Accept-Encoding": "gzip, deflate",
#   "Authorization": "Basic NDZiNjMxNDQwNTMxZGJkYjU2OTMwYjJlZWFjODc0NGE6ZGRlMzQzNjBjY2NlZjAwYzVmNzM3OWI1ODQ3MzdhNTQ=",
#   "Connection": "keep-alive",
#   "Content-Type": "application/json"
# }

# response = requests.request("POST", url, headers=headers, data=json.dumps(payload))



import requests
import json

url = "https://apps.beam.cloud/0oapt"
payload = {"text": "Helin says hello again!"}
headers = {
  "Accept": "*/*",
  "Accept-Encoding": "gzip, deflate",
  "Authorization": "Basic NDZiNjMxNDQwNTMxZGJkYjU2OTMwYjJlZWFjODc0NGE6ZGRlMzQzNjBjY2NlZjAwYzVmNzM3OWI1ODQ3MzdhNTQ=",
  "Connection": "keep-alive",
  "Content-Type": "application/json"
}

response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

print(response.text)