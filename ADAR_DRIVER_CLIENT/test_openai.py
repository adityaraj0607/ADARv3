"""
Test OpenAI API Key from .env file
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

print("=" * 60)
print("OpenAI API Key Test")
print("=" * 60)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key exists
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file!")
    print("\nMake sure you have a .env file with:")
    print("OPENAI_API_KEY=sk-...")
    exit(1)

print(f"✅ API Key loaded from .env")
print(f"   Key preview: {api_key[:15]}...{api_key[-4:]}")
print(f"   Key length: {len(api_key)} characters")
print()

# Test API connection
print("Testing API connection...")
try:
    client = OpenAI(api_key=api_key)
    
    # Test with GPT-4o (the model ADAR uses)
    print("Sending test request to GPT-4o...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Respond with exactly: 'ADAR API test successful!'"}
        ],
        max_tokens=20
    )
    
    print()
    print("=" * 60)
    print("✅ SUCCESS! OpenAI API is working correctly!")
    print("=" * 60)
    print(f"Response: {response.choices[0].message.content}")
    print()
    print("Token Usage:")
    print(f"  - Prompt tokens: {response.usage.prompt_tokens}")
    print(f"  - Completion tokens: {response.usage.completion_tokens}")
    print(f"  - Total tokens: {response.usage.total_tokens}")
    print()
    print("Approximate cost for this test call:")
    # GPT-4o pricing (as of 2024): $5/1M input, $15/1M output
    input_cost = (response.usage.prompt_tokens / 1_000_000) * 5
    output_cost = (response.usage.completion_tokens / 1_000_000) * 15
    total_cost = input_cost + output_cost
    print(f"  ~${total_cost:.6f} USD")
    print()
    print("Your API key is working correctly! ✅")
    print("ADAR system can now use JARVIS (GPT-4o) features.")
    
except Exception as e:
    print()
    print("=" * 60)
    print("❌ ERROR: API call failed!")
    print("=" * 60)
    print(f"Error: {e}")
    print()
    
    if "invalid_api_key" in str(e).lower():
        print("Issue: Invalid API key")
        print("Solution: Check your .env file and make sure the key is correct")
    elif "insufficient_quota" in str(e).lower():
        print("Issue: Insufficient quota or billing")
        print("Solution: Add credits to your OpenAI account at:")
        print("  https://platform.openai.com/account/billing")
    elif "rate_limit" in str(e).lower():
        print("Issue: Rate limit exceeded")
        print("Solution: Wait a moment and try again")
    else:
        print("Check your internet connection and API key")
    
    print()
    print("Visit https://platform.openai.com/api-keys to verify your key")
    exit(1)

print()
print("You can now run your ADAR system with confidence!")
print("=" * 60)
