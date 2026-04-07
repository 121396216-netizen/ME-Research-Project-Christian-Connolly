#!/usr/bin/env python3
import requests, json

resp = requests.post('http://127.0.0.1:8081/v1/chat/completions', json={
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50,
    "temperature": 0
})
print(json.dumps(resp.json(), indent=2))
