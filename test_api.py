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
        ("/api/settings", "GET"),
        ("/api/start", "POST"),
        ("/api/stop", "POST")
    ]
    
    # Common headers to test with
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Origin': 'http://localhost:5000'  # Simulate browser origin
    }
    
    results = {}
    
    for endpoint, method in endpoints:
        full_url = f"{base_url}{endpoint}"
        print(f"\nTesting {method} {full_url}...")
        
        try:
            if method == "GET":
                response = requests.get(full_url, headers=headers, timeout=3)
            elif method == "POST":
                response = requests.post(full_url, headers=headers, json={}, timeout=3)
            
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"Response data: {json.dumps(data, indent=2)}")
                    results[endpoint] = {"success": True, "data": data}
                except json.JSONDecodeError:
                    print(f"Error: Response is not valid JSON: {response.text}")
                    results[endpoint] = {"success": False, "error": "Invalid JSON response"}
            else:
                print(f"Error response: {response.text}")
                results[endpoint] = {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"Error calling {full_url}: {str(e)}")
            results[endpoint] = {"success": False, "error": str(e)}
    
    return results

def test_cors_preflight(base_url="http://localhost:5000"):
    """Test CORS preflight requests specifically."""
    print("\nTesting CORS preflight (OPTIONS) requests...")
    
    endpoints = [
        "/api/status",
        "/api/calibration/status",
        "/api/settings",
        "/api/start",
        "/api/stop"
    ]
    
    headers = {
        'Origin': 'http://example.com',  # Different origin to test CORS
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'Content-Type'
    }
    
    for endpoint in endpoints:
        full_url = f"{base_url}{endpoint}"
        print(f"\nTesting OPTIONS {full_url}...")
        
        try:
            response = requests.options(full_url, headers=headers, timeout=3)
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            # Check for CORS headers
            cors_headers = {k: v for k, v in response.headers.items() if k.lower().startswith('access-control')}
            if cors_headers:
                print(f"CORS headers present: {cors_headers}")
            else:
                print("Warning: No CORS headers found in response")
                
        except Exception as e:
            print(f"Error calling OPTIONS {full_url}: {str(e)}")

if __name__ == "__main__":
    # Use provided port or default
    port = sys.argv[1] if len(sys.argv) > 1 else "5000"
    base_url = f"http://localhost:{port}"
    
    # Test regular API endpoints
    test_api(base_url)
    
    # Test CORS preflight
    test_cors_preflight(base_url)