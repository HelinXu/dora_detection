
import requests
import json
import base64
import cv2
import numpy as np
import time

image_name = 'example.jpeg'

# encode image base64
retval, buffer = cv2.imencode('.jpg', cv2.imread(image_name))
text = str(base64.b64encode(buffer).decode('utf-8'))

# sent the image to the app on web
url = "https://apps.beam.cloud/yt8b8"
payload = {"image_base64": text}
headers = {
  "Accept": "*/*",
  "Accept-Encoding": "gzip, deflate",
  "Authorization": "Basic NDZiNjMxNDQwNTMxZGJkYjU2OTMwYjJlZWFjODc0NGE6ZGRlMzQzNjBjY2NlZjAwYzVmNzM3OWI1ODQ3MzdhNTQ=",
  "Connection": "keep-alive",
  "Content-Type": "application/json"
}

tic = time.time()

response = requests.request("POST", url, 
  headers=headers,
  data=json.dumps(payload)
)
toc = time.time()

print("time:", toc - tic)

# save the returned results
response = response.json()

jpg_as_text = bytes(response['response'], 'utf-8')

jpg_original = base64.b64decode(jpg_as_text)

image = cv2.imdecode(np.frombuffer(jpg_original, dtype=np.uint8), -1)

# save image
cv2.imwrite('outputt.png', image)