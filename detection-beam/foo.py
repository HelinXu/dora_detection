
import requests
import json
import base64
import cv2
import numpy as np

image_name = 'test.jpg'

# encode image base64
retval, buffer = cv2.imencode('.jpg', cv2.imread(image_name))
text = str(base64.b64encode(buffer).decode('utf-8'))

print(text)

# print(type(text))

# jpg_as_text = bytes(text, 'utf-8')

# # decode image base64
# jpg_original = base64.b64decode(jpg_as_text)

# image = cv2.imdecode(np.frombuffer(jpg_original, dtype=np.uint8), -1)

# # save image
# cv2.imwrite('output.png', image)

url = "https://apps.beam.cloud/yt8b8"
payload = {"image_base64": text}
headers = {
  "Accept": "*/*",
  "Accept-Encoding": "gzip, deflate",
  "Authorization": "Basic NDZiNjMxNDQwNTMxZGJkYjU2OTMwYjJlZWFjODc0NGE6ZGRlMzQzNjBjY2NlZjAwYzVmNzM3OWI1ODQ3MzdhNTQ=",
  "Connection": "keep-alive",
  "Content-Type": "application/json"
}

response = requests.request("POST", url, 
  headers=headers,
  data=json.dumps(payload)
)

print(response.text)