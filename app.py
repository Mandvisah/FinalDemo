from flask import Flask, render_template, request, jsonify, session
import csv
import os
import time
from datetime import datetime
from verify import KeystrokeVerifier

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change in production

# Initialize verifier (will be loaded on first use)
verifier = None

def get_verifier():
    global verifier
    if verifier is None:
        try:
            verifier = KeystrokeVerifier()
        except Exception as e:
            print(f"Warning: Could not load verifier: {e}")
            verifier = None
    return verifier

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record_keystrokes():
    data = request.json
    keystrokes = data.get("keystrokes", [])
    user_id = data.get("user_id", "unknown")

    if not keystrokes:
        return jsonify({"status": "error", "message": "No keystrokes received"}), 400

    # Find next available file number
    i = 1
    while os.path.exists(f"./data/sample{i}.csv"):
        i += 1
    
    filename = f"./data/sample{i}.csv"
    
    try:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keystrokes[0].keys())
            writer.writeheader()
            writer.writerows(keystrokes)
        
        # Verify immediately after recording
        verification_result = None
        verifier_instance = get_verifier()
        if verifier_instance:
            verification_result = verifier_instance.verify_attempt(filename)
        
        response = {
            "status": "success", 
            "message": f"Keystrokes saved as sample{i}.csv",
            "file_id": i
        }
        
        if verification_result:
            response["verification"] = {
                "authentic": verification_result['ensemble_prediction'] == 'Legitimate',
                "confidence": verification_result['ensemble_confidence'],
                "security_level": verification_result['security_level']
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify_keystrokes():
    data = request.json
    keystrokes = data.get("keystrokes", [])
    
    if not keystrokes:
        return jsonify({"status": "error", "message": "No keystrokes received"}), 400
    
    # Save to temporary file for verification
    temp_file = f"./data/temp_verify_{int(time.time())}.csv"
    
    try:
        with open(temp_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keystrokes[0].keys())
            writer.writeheader()
            writer.writerows(keystrokes)
        
        # Verify the attempt
        verifier_instance = get_verifier()
        if not verifier_instance:
            return jsonify({"status": "error", "message": "Verification system not available"}), 500
        
        result = verifier_instance.verify_attempt(temp_file)
        
        # Clean up temp file
        os.remove(temp_file)
        
        if result:
            return jsonify({
                "status": "success",
                "authentic": result['ensemble_prediction'] == 'Legitimate',
                "confidence": result['ensemble_confidence'],
                "security_level": result['security_level'],
                "anomaly_score": result['anomaly_score'],
                "details": {
                    "keystrokes": result['keystrokes'],
                    "dwell_time_avg": result['features']['dwell_mean'],
                    "flight_time_avg": result['features']['flight_mean']
                }
            })
        else:
            return jsonify({"status": "error", "message": "Verification failed"}), 500
            
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/status')
def status():
    """Check system status"""
    verifier_instance = get_verifier()
    model_loaded = verifier_instance is not None
    
    return jsonify({
        "status": "online",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)