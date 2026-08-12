"""
Senate AI - AI Client
Uses Puter.js free AI models - no API keys needed.
"""

import requests
import time

PUTER_URL = "https://api.puter.com/v1/chat/completions"

def call_ai(prompt, max_tokens=500, model="gpt-4o-mini"):
    """Call Puter.js API - completely free, no API key needed"""
    
    headers = {
        "Content-Type": "application/json",
        "Origin": "https://puter.com",
        "Referer": "https://puter.com/"
    }
    
    for attempt in range(3):
        try:
            r = requests.post(
                PUTER_URL,
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            
            print(f"   API {r.status_code}: {r.text[:100]}")
            time.sleep(3)
        
        except Exception as e:
            print(f"   Attempt {attempt+1}: {e}")
            time.sleep(3)
    
    return None
