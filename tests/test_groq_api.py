import os
import sys
from groq import Groq

# Use dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def test_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("FAIL: GROQ_API_KEY not found in environment.")
        return
    
    print(f"Testing Groq API with key starting with: {api_key[:10]}...")
    
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello, respond with only the word 'Antigravity' if the API is working."}]
        )
        response = completion.choices[0].message.content.strip()
        print(f"Response: {response}")
        if "Antigravity" in response:
            print("SUCCESS: Groq setup works perfectly!")
        else:
            print(f"WARNING: API responded, but content was unusual: {response}")
            
    except Exception as e:
        print(f"FAIL: Groq API call failed: {e}")

if __name__ == "__main__":
    test_groq()
