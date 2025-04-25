#!/usr/bin/env python3
import requests
import json
import time
import sys

def test_api(base_url="http://localhost:5000"):
    """Test the API endpoints and report results."""
    print(f"Testing API at {base_url}...")
    
    endpoints = [
        ("/api/status", "GET"),
        ("/api/calibration/status", "GET"),
        ("/api/settings", "GET")
    ]
    
    results = {}
    
    for endpoint, method in endpoints:
        full_url = f"{base_url}{endpoint}"
        print(f"\nTesting {method} {full_url}...")
        
        try:
            if method == "GET":
                response = requests.get(full_url, timeout=3)
            elif method == "POST":
                response = requests.post(full_url, json={}, timeout=3)
            
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response data: {json.dumps(data, indent=2)}")
                results[endpoint] = {"success": True, "data": data}
            else:
                print(f"Error response: {response.text}")
                results[endpoint] = {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"Error calling {full_url}: {str(e)}")
            results[endpoint] = {"success": False, "error": str(e)}
    
    return results

if __name__ == "__main__":
    # Use provided port or default
    port = sys.argv[1] if len(sys.argv) > 1 else "5000"
    test_api(f"http://localhost:{port}")