"""No SDK needed. Run after starting LoopForge on 127.0.0.1:8787."""
import json
import os
import urllib.request

payload = {"model": "loopforge", "messages": [{"role": "user", "content": "Inspect the repository and explain its entry point."}],
           "mode": "agent", "stream": False}
request = urllib.request.Request("http://127.0.0.1:8787/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("LOOPFORGE_SERVER_KEY", "local")})
with urllib.request.urlopen(request, timeout=1300) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
print("Session:", result["x_loopforge"]["session_id"])
