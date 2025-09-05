import os
import dspy
import requests
try:
    from dotenv import load_dotenv
    # Load .env file but don't override existing environment variables
    load_dotenv(override=False)
except ImportError:
    pass

try:
    import lmstudio as lms
    HAS_LMSTUDIO_SDK = True
except ImportError:
    HAS_LMSTUDIO_SDK = False

def check_ollama_connection():
    """Check if Ollama API is accessible via DSPy"""
    try:
        # Get API base from environment
        api_base = os.getenv('OLLAMA_API_BASE', 'http://localhost:1234/v1')
        
        # Test basic connectivity
        response = requests.get(f"{api_base.replace('/v1', '')}/api/tags", timeout=5)
        print(f"API endpoint reachable: {response.status_code == 200}")
        
        # Test DSPy integration
        lm = dspy.LM(
            model='qwen/qwen3-coder-30b',
            base_url=api_base
        )
        
        result = lm("Test connection. Respond with 'OK' if you receive this.")
        print(f"DSPy Ollama response: {result}")
        return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def check_lmstudio_connection():
    """Check if LMStudio API is accessible via DSPy"""
    try:
        # Get API base from environment, defaulting to common LMStudio port
        api_base = os.getenv('OLLAMA_API_BASE', 'http://localhost:1234/v1')
        
        # Test basic connectivity to LMStudio
        # LMStudio uses OpenAI-compatible API, so test /models endpoint
        models_url = f"{api_base}/models"
        response = requests.get(models_url, timeout=5)
        print(f"LMStudio API endpoint reachable: {response.status_code == 200}")
        
        loaded_models = []
        if response.status_code == 200:
            models = response.json().get('data', [])
            print(f"Available models: {len(models)}")
            if models:
                print(f"Model names: {[m.get('id', 'unknown') for m in models[:3]]}")
        
        # Use LM Studio SDK to get loaded models if available
        if HAS_LMSTUDIO_SDK:
            try:
                # Get SERVER_API_HOST from environment or parse from api_base
                server_host = os.getenv('SERVER_API_HOST')
                if not server_host:
                    from urllib.parse import urlparse
                    parsed = urlparse(api_base.replace('/v1', ''))
                    server_host = f"{parsed.hostname or 'localhost'}:{parsed.port or 1234}"
                
                lms.configure_default_client(server_host)
                loaded_models_info = lms.list_loaded_models()
                loaded_models = [model.identifier for model in loaded_models_info]
                print(f"✓ Loaded models (via SDK): {loaded_models}")
            except Exception as e:
                print(f"SDK method failed: {e}")
                # Fallback: assume models from /models endpoint are loaded
                if response.status_code == 200:
                    loaded_models = [m.get('id') for m in models]
                    print(f"Fallback: assuming available models are loaded")
        else:
            print("LM Studio SDK not available, assuming available models are loaded")
            if response.status_code == 200:
                loaded_models = [m.get('id') for m in models]
        
        print(f"Models to test: {loaded_models[:3]}")
        
        # Test DSPy integration with LMStudio
        # Prefer a loaded model, fallback to first available
        model_name = 'gpt-3.5-turbo'  # LMStudio default
        if loaded_models:
            model_name = loaded_models[0]  # Use first loaded model
        elif response.status_code == 200:
            models = response.json().get('data', [])
            if models:
                model_name = models[0].get('id', model_name)
        
        print(f"Using model: {model_name}")
        
        # Configure DSPy for LMStudio (OpenAI-compatible endpoint)
        lm = dspy.LM(
            model=f"openai/{model_name}",  # Need provider for litellm backend
            base_url=api_base,
            api_key="dummy"  # LMStudio doesn't require a real API key
        )
        
        print(f"Sending request to {api_base} with model openai/{model_name}")
        result = lm("Test connection. Can you hear me?")
        print(f"DSPy LMStudio response: {result}")
        print(f"Response type: {type(result)}, Length: {len(str(result))}")
        return True
        
    except Exception as e:
        print(f"LMStudio connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Ollama connection...")
    # ollama_success = check_ollama_connection()
    
    print("\nTesting LMStudio connection...")
    lmstudio_success = check_lmstudio_connection()
    
    print(f"\nResults:")
    # print(f"Ollama: {'✓' if ollama_success else '✗'}")
    print(f"LMStudio: {'✓' if lmstudio_success else '✗'}")