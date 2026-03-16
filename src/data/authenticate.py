import requests
import json

reqUrl = "https://kystdatahuset.no/ws/api/auth/login"

headersList = {
 "User-Agent": "Your Client (https://your-client.com)",
 "accept": "*/*",
 "Content-Type": "application/json" 
}

payload = json.dumps({
  "username": "username",
  "password": """PASSWORD"""
})

response = requests.request("POST", reqUrl, data=payload,  headers=headersList)

print(response.text)