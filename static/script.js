let keystrokes = [];
let pressTimes = {};
let startTime = null;
const passwordInput = document.getElementById("passwordInput");
let isRecording = false;

passwordInput.addEventListener("focus", () => {
    if (!startTime) {
        startTime = performance.now();
        isRecording = true;
        keystrokes = [];
        pressTimes = {};
    }
});

passwordInput.addEventListener("keydown", (event) => {
    if (!isRecording) return;

    const key = event.key;
    const currentTime = performance.now() - startTime;

    // Ignore modifier keys and special keys
    if (key === "Shift" || key === "Control" || key === "Alt" || key === "Meta") {
        return;
    }

    // If Enter is pressed, submit instead of creating newline
    if (key === "Enter") {
        event.preventDefault();
        sendData();
        return;
    }

    // Record press time
    if (!pressTimes[key]) {
        pressTimes[key] = currentTime;
    }
});

passwordInput.addEventListener("keyup", (event) => {
    if (!isRecording) return;

    const key = event.key;
    const currentTime = performance.now() - startTime;

    if (pressTimes[key] !== undefined) {
        const dwell = currentTime - pressTimes[key];
        
        const keystrokeData = {
            key: key,
            press_time: parseFloat(pressTimes[key].toFixed(3)),
            release_time: parseFloat(currentTime.toFixed(3)),
            dwell_time: parseFloat(dwell.toFixed(3)),
            flight_time: 0.0
        };

        // Compute flight time for previous key
        if (keystrokes.length > 0) {
            const prev = keystrokes[keystrokes.length - 1];
            keystrokes[keystrokes.length - 1].flight_time = parseFloat(
                (keystrokeData.press_time - prev.release_time).toFixed(3)
            );
        }

        keystrokes.push(keystrokeData);
        delete pressTimes[key];
    }
});

function startAuthentication() {
    if (keystrokes.length === 0) {
        alert("Please type your password first!");
        return;
    }

    showLoading(true);
    
    // Use the verification endpoint for real-time authentication
    fetch("/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            keystrokes: keystrokes
        })
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        displayResults(data);
    })
    .catch(error => {
        showLoading(false);
        console.error("Error:", error);
        alert("Authentication failed. Please try again.");
    });
}

function displayResults(data) {
    const resultsDiv = document.getElementById("results");
    const authResult = document.getElementById("authResult");
    const resultText = document.getElementById("resultText");
    const confidenceValue = document.getElementById("confidenceValue");
    const securityLevel = document.getElementById("securityLevel");
    const anomalyScore = document.getElementById("anomalyScore");
    const keystrokeCount = document.getElementById("keystrokeCount");

    resultsDiv.style.display = "block";

    if (data.status === "success") {
        keystrokeCount.textContent = data.details.keystrokes;
        confidenceValue.textContent = data.confidence.toFixed(1) + "%";
        anomalyScore.textContent = data.anomaly_score.toFixed(1) + "%";
        
        // Set security level with color
        securityLevel.textContent = data.security_level;
        securityLevel.className = "stat-value security-" + data.security_level.toLowerCase();

        if (data.authentic) {
            authResult.className = "result-item status-authentic";
            resultText.innerHTML = "✅ <strong>Authentication Successful!</strong><br>Your typing pattern matches the enrolled user.";
        } else {
            authResult.className = "result-item status-impostor";
            resultText.innerHTML = "❌ <strong>Authentication Failed!</strong><br>Your typing pattern does not match.";
        }
    } else {
        authResult.className = "result-item status-impostor";
        resultText.textContent = "Error: " + data.message;
    }
}

function showLoading(show) {
    const loadingDiv = document.getElementById("loading");
    const button = document.querySelector(".btn-primary");
    
    if (show) {
        loadingDiv.style.display = "block";
        button.disabled = true;
        button.textContent = "Processing...";
    } else {
        loadingDiv.style.display = "none";
        button.disabled = false;
        button.textContent = "🔒 Authenticate";
    }
}

function clearInput() {
    passwordInput.value = "";
    keystrokes = [];
    pressTimes = {};
    startTime = null;
    isRecording = false;
    document.getElementById("results").style.display = "none";
}

// Also save data for training purposes
function sendData() {
    if (keystrokes.length === 0) {
        alert("No keystrokes recorded!");
        return;
    }

    showLoading(true);

    fetch("/record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            keystrokes: keystrokes,
            user_id: "current_user"  // In real system, this would be the logged-in user
        })
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        if (data.verification) {
            displayResults({
                status: "success",
                authentic: data.verification.authentic,
                confidence: data.verification.confidence,
                security_level: data.verification.security_level,
                anomaly_score: 0,
                details: { keystrokes: keystrokes.length }
            });
        }
        alert("Data saved: " + data.message);
    })
    .catch(error => {
        showLoading(false);
        console.error("Error:", error);
        alert("Failed to save data.");
    });
}

// Check system status on page load
window.addEventListener('load', () => {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            if (!data.model_loaded) {
                console.warn('Authentication model not loaded');
            }
        });
});