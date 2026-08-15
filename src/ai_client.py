"""
Senate AI - AI Client (Python wrapper)
Calls Node.js/Puter.js for AI - no API key needed.
"""

import subprocess
import json
import tempfile
import os

def call_ai(prompt, max_tokens=500, model="gpt-4o-mini"):
    """Call Puter.js AI through Node.js"""
    
    try:
        result = subprocess.run(
            ['node', 'src/ai_client.js'],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        
        return None
    
    except Exception as e:
        print(f"   AI call failed: {e}")
        return None
