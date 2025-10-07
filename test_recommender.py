import requests
import uuid
import json
import time

API_URL = "http://localhost:8000/chat"   # change to your deployed endpoint
HEADERS = {"Content-Type": "application/json"}

def send_request(session_id: str, message: str):
    """Send one chat message and return parsed response."""
    payload = {"session_id": session_id, "message": message}
    start = time.time()
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    elapsed = time.time() - start

    print(f"\n📡 Sent: {message}")
    print(f"⏱️  Latency: {elapsed:.2f}s, HTTP {r.status_code}")

    if r.status_code != 200:
        print("❌ Error:", r.text)
        return None

    data = r.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    # Minimal schema sanity check
    assert "recommendations" in data, "Missing recommendations"
    if data["recommendations"]:
        first = data["recommendations"][0]
        print(f"✅ Top pick: {first['title']} — ${first['price']}")

    return data


def run_scenarios():
    # Unique session id to persist profile across turns
    session_id = f"iot-{uuid.uuid4().hex[:8]}"

    print("🔧 Scenario 1: Android, camera-first, <$900")
    resp1 = send_request(session_id,
        "Looking for Android phones under $900 with long battery life and good camera")
    time.sleep(2)

    if resp1 and resp1["recommendations"]:
        # pick first phone id
        pick_id = resp1["recommendations"][0]["id"]
        print(f"🎯 Asking bundle for {pick_id}")
        send_request(session_id, f"I pick {pick_id}. Give me accessories under $100.")

    print("\n🍎 Scenario 2: iOS user with MagSafe preference")
    send_request(f"iot-{uuid.uuid4().hex[:8]}",
        "Prefer iOS phones with MagSafe under 1400 SGD, best camera priority 5.")

    print("\n🎮 Scenario 3: Gaming phone preference")
    send_request(f"iot-{uuid.uuid4().hex[:8]}",
        "I want an Android phone good for gaming and performance under 1200 dollars.")

    print("\n⚙️ Scenario 4: Update user profile explicitly (IoT-style config)")
    send_request(session_id,
        "Update my profile: prefer Android, budget 1000, camera priority 5, gaming 4, accessories interest charger and case.")


if __name__ == "__main__":
    try:
        run_scenarios()
        print("\n✅ All IoT test scenarios completed.\n")
    except AssertionError as e:
        print("❌ Test failed:", e)
    except Exception as e:
        print("⚠️ Unexpected error:", e)
