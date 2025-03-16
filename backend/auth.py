import yaml
import requests

def load_config():
    """Load configuration from config.yaml file"""
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def test_auth():
    """Test authentication with AnythingLLM server"""
    config = load_config()
    
    # Construct the auth URL
    auth_url = f"{config['model_server_base_url']}/auth"
    
    # Set up the headers with API key
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }
    
    # Make the request
    response = requests.get(auth_url, headers=headers)
    
    # Check if authentication was successful
    if response.status_code == 200:
        print("✅ Authentication successful")
        return True
    else:
        print(f"❌ Authentication failed: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    # Test authentication when the script is run directly
    test_auth()
