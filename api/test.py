from PIL import Image
import requests
from io import BytesIO
import json
import requests
url = "http://127.0.0.1:8000/image/video/"

payload={'lable_status': 'dsad'}
files=[
  ('filefiled',('2022-09-22_00-17-45.mkv',open('D:/HP/Documents/pothole/data/detect/project/detect code/video.mp4','rb'),'application/octet-stream'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files).json()




ss = requests.get(response['msg']['filefiled'])
print(ss)


