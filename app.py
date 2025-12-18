from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import csv
import os
import time
from datetime import datetime
from verify import KeystrokeVerifier
from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change in production

# MongoDB Configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['keystroke_auth_db']
keystrokes_collection = db['keystrokes']
users_collection = db['users']

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
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = users_collection.find_one({'email': email})
        
        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['user_email'] = user['email']
            session['user_name'] = user['name']
            session['user_id'] = str(user['_id'])
            return redirect(url_for('dashboard'))
            
        return render_template('login.html', error="Invalid email or password")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if users_collection.find_one({'email': email}):
            return render_template('register.html', error="Email already exists")
            
        hashed_password = generate_password_hash(password)
        
        user_data = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.now()
        }
        
        result = users_collection.insert_one(user_data)
        
        session['logged_in'] = True
        session['user_email'] = email
        session['user_name'] = name
        session['user_id'] = str(result.inserted_id)
        
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/record', methods=['POST'])
def record_keystrokes():
    data = request.json
    keystrokes = data.get("keystrokes", [])
    user_id = session.get('user_id') or data.get("user_id", "unknown")

    if not keystrokes:
        return jsonify({"status": "error", "message": "No keystrokes received"}), 400

    # Find next available file number
    i = 1
    while os.path.exists(f"./data/sample{i}.csv"):
        i += 1
    
    filename = f"./data/sample{i}.csv"
    
    try:
        # Save to MongoDB
        mongo_record = {
            "user_id": user_id,
            "timestamp": datetime.now(),
            "keystrokes": keystrokes
        }
        inserted_id = keystrokes_collection.insert_one(mongo_record).inserted_id

        # Save to CSV (required for current verifier implementation)
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
            "message": f"Keystrokes saved to MongoDB (ID: {str(inserted_id)}) and CSV (sample{i}.csv)",
            "file_id": i,
            "mongo_id": str(inserted_id)
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
    user_id = session.get('user_id', 'anonymous') # Get user_id from session if available
    
    if not keystrokes:
        return jsonify({"status": "error", "message": "No keystrokes received"}), 400
    
    # Save to MongoDB for logging/audit
    try:
        mongo_record = {
            "user_id": user_id,
            "timestamp": datetime.now(),
            "keystrokes": keystrokes,
            "type": "verification_attempt"
        }
        keystrokes_collection.insert_one(mongo_record)
    except Exception as e:
        print(f"Warning: Failed to save to MongoDB: {e}")

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