import requests

def test_validation():
    url = "http://localhost:5000/api/crop-recommend"
    
    # Test valid input
    valid_payload = {
        "N": 90, "P": 42, "K": 43, 
        "temperature": 20.8, "humidity": 82, 
        "ph": 6.5, "rainfall": 202.9
    }
    print("Testing valid input...")
    r = requests.post(url, json=valid_payload)
    print(f"Status: {r.status_code}, Response: {r.json()}\n")

    # Test invalid Nitrogen (too high)
    invalid_payload = valid_payload.copy()
    invalid_payload["N"] = 500
    print("Testing invalid N (500, max 140)...")
    r = requests.post(url, json=invalid_payload)
    print(f"Status: {r.status_code}, Response: {r.json()}\n")

    # Test invalid Rainfall (too low)
    invalid_payload = valid_payload.copy()
    invalid_payload["rainfall"] = 5
    print("Testing invalid Rainfall (5, min 20)...")
    r = requests.post(url, json=invalid_payload)
    print(f"Status: {r.status_code}, Response: {r.json()}\n")

if __name__ == "__main__":
    test_validation()
