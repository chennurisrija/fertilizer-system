from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3, hashlib, json, os
from datetime import datetime
import joblib
import numpy as np

app = Flask(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

DB = os.path.join(os.path.dirname(__file__), 'fertilizer.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crop_model.pkl')

# ── Load trained ML model ─────────────────────────────────────────────────
MODEL = joblib.load(MODEL_PATH)

# ── Database setup ────────────────────────────────────────────────────────
def init_db():
    c = sqlite3.connect(DB)
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        crop TEXT,
        nitrogen REAL,
        phosphorus REAL,
        potassium REAL,
        ph REAL,
        soil_type TEXT,
        moisture REAL,
        location TEXT,
        result TEXT,
        predicted_crop TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    c.commit()
    c.close()

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def sha(p):
    return hashlib.sha256(p.encode()).hexdigest()

# ── NPK crop database for baseline recommendation ─────────────────────────
CROPS = {
    "rice":{"N":(80,120),"P":(40,60),"K":(40,60)},
    "wheat":{"N":(60,100),"P":(30,50),"K":(30,50)},
    "maize":{"N":(100,140),"P":(50,70),"K":(50,70)},
    "corn":{"N":(100,140),"P":(50,70),"K":(50,70)},
    "sugarcane":{"N":(150,200),"P":(60,80),"K":(80,100)},
    "cotton":{"N":(80,120),"P":(40,60),"K":(60,80)},
    "tomato":{"N":(60,100),"P":(50,70),"K":(80,100)},
    "potato":{"N":(100,140),"P":(60,80),"K":(80,120)},
    "onion":{"N":(60,80),"P":(40,60),"K":(60,80)},
    # Add more crops as needed
}
DEFAULT_NPK = {"N":(60,100),"P":(40,60),"K":(50,70)}

SOIL_ADJ = {
    "Sandy":{"N":1.2,"P":1.1,"K":1.3},
    "Clay":{"N":0.9,"P":0.9,"K":0.8},
    "Loamy":{"N":1.0,"P":1.0,"K":1.0},
    "Silty":{"N":1.0,"P":1.1,"K":0.9},
    "Peaty":{"N":0.8,"P":1.2,"K":1.1},
    "Chalky":{"N":1.1,"P":0.9,"K":1.0},
}

def nutrient_status(v, low, mid):
    if v < low: return "Low"
    if v < mid: return "Medium"
    return "High"

# ── Compute soil/fertilizer recommendation ────────────────────────────────
def compute_recommendation(crop, N, P, K, ph, soil_type):
    key = crop.strip().lower()
    req = CROPS.get(key, DEFAULT_NPK)
    unknown = key not in CROPS
    adj = SOIL_ADJ.get(soil_type, {"N":1.0,"P":1.0,"K":1.0})

    tN = ((req["N"][0] + req["N"][1]) / 2) * adj["N"]
    tP = ((req["P"][0] + req["P"][1]) / 2) * adj["P"]
    tK = ((req["K"][0] + req["K"][1]) / 2) * adj["K"]

    dN = max(0.0, tN - N)
    dP = max(0.0, tP - P)
    dK = max(0.0, tK - K)

    if ph < 5.5:   ph_advice = "Very acidic soil. Apply agricultural lime to raise pH."
    elif ph < 6.0: ph_advice = "Slightly acidic. Consider applying lime."
    elif ph > 8.0: ph_advice = "Very alkaline. Apply sulfur or gypsum to lower pH."
    elif ph > 7.5: ph_advice = "Slightly alkaline. Use acidic fertilizers."
    else:          ph_advice = "pH is in the optimal range for most crops."

    fertilizers = []
    rN, rP, rK = dN, dP, dK

    if rP > 5:
        dap_kg = round((rP / 46.0) * 100, 1)
        n_from_dap = dap_kg * 0.18
        rN = max(0.0, rN - n_from_dap)
        fertilizers.append({
            "name": "DAP (Di-ammonium Phosphate)",
            "quantity_kg": dap_kg,
            "purpose": "Provides Phosphorus + some Nitrogen",
            "cost": round(dap_kg * 27)
        })

    if rK > 5:
        mop_kg = round((rK / 60.0) * 100, 1)
        fertilizers.append({
            "name": "MOP (Muriate of Potash)",
            "quantity_kg": mop_kg,
            "purpose": "Provides Potassium",
            "cost": round(mop_kg * 17)
        })

    if rN > 5:
        urea_kg = round((rN / 46.0) * 100, 1)
        fertilizers.append({
            "name": "Urea",
            "quantity_kg": urea_kg,
            "purpose": "Provides Nitrogen",
            "cost": round(urea_kg * 5.5)
        })

    total_cost = sum(f["cost"] for f in fertilizers)
    avg_cost   = round(total_cost * 1.45)
    savings    = round(avg_cost - total_cost)
    pct        = round(savings / avg_cost * 100, 1) if avg_cost > 0 else 0

    organics = []
    if dN > 10: organics.append({"name":"Vermicompost","qty":"200-300 kg/acre","benefit":"Slow-release nitrogen source"})
    if dP > 10: organics.append({"name":"Bone Meal","qty":"50-80 kg/acre","benefit":"Natural phosphorus source"})
    if dK > 10: organics.append({"name":"Wood Ash","qty":"100-150 kg/acre","benefit":"Natural potassium source"})
    if not organics:
        organics.append({"name":"Compost","qty":"200 kg/acre","benefit":"General soil health improvement"})

    first_fert = fertilizers[0]["name"] if fertilizers else "basal fertilizer"

    return {
        "crop": crop.title(),
        "soil_type": soil_type,
        "unknown_crop": unknown,
        "unknown_note": ("'{}' not in database. Using general crop standards.".format(crop.title())) if unknown else "",
        "ph_advice": ph_advice,
        "current": {
            "nitrogen":   {"value": N, "status": nutrient_status(N, 40, 80)},
            "phosphorus": {"value": P, "status": nutrient_status(P, 20, 50)},
            "potassium":  {"value": K, "status": nutrient_status(K, 30, 80)},
            "ph":         {"value": ph}
        },
        "required": {
            "nitrogen":   round(tN, 1),
            "phosphorus": round(tP, 1),
            "potassium":  round(tK, 1)
        },
        "deficit": {
            "nitrogen":   round(dN, 1),
            "phosphorus": round(dP, 1),
            "potassium":  round(dK, 1)
        },
        "fertilizers": fertilizers,
        "organics": organics,
        "cost": {
            "estimated": total_cost,
            "without":   avg_cost,
            "savings":   savings,
            "percent":   pct
        },
        "tips": [
            "Apply {} at the time of sowing for best results.".format(first_fert),
            "Split nitrogen application: 50% basal dose + 50% top-dress at tillering.",
            "Irrigate lightly before applying potash for better absorption.",
            "Test soil again every 2 crop seasons for accurate recommendations.",
            "Maintain soil moisture at 40-60% for optimal nutrient uptake by roots."
        ]
    }

# ── Serve frontend ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)

# ── API routes ─────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

@app.route('/api/register', methods=['POST'])
def register():
    d = request.get_json(force=True) or {}
    name  = str(d.get('name','')).strip()
    email = str(d.get('email','')).strip().lower()
    pwd   = str(d.get('password',''))
    if not name or not email or not pwd:
        return jsonify({"error": "All fields are required"}), 400
    if len(pwd) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    conn = get_db()
    try:
        conn.execute("INSERT INTO users(name,email,password) VALUES(?,?,?)", (name, email, sha(pwd)))
        conn.commit()
        u = conn.execute("SELECT id,name,email FROM users WHERE email=?", (email,)).fetchone()
        return jsonify({"ok": True, "user": {"id": u["id"], "name": u["name"], "email": u["email"]}})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered"}), 409
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    d = request.get_json(force=True) or {}
    email = str(d.get('email','')).strip().lower()
    pwd   = str(d.get('password',''))
    conn  = get_db()
    u = conn.execute("SELECT id,name,email FROM users WHERE email=? AND password=?", (email, sha(pwd))).fetchone()
    conn.close()
    if not u:
        return jsonify({"error": "Invalid email or password"}), 401
    return jsonify({"ok": True, "user": {"id": u["id"], "name": u["name"], "email": u["email"]}})

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    d = request.get_json(force=True) or {}
    try:
        crop      = str(d.get('crop', '')).strip()
        N         = float(d.get('nitrogen',  0))
        P         = float(d.get('phosphorus', 0))
        K         = float(d.get('potassium', 0))
        ph        = float(d.get('ph', 7.0))
        soil_type = str(d.get('soil_type', 'Loamy'))
        moisture  = float(d.get('moisture', 50))
        location  = str(d.get('location', ''))
        user_id   = int(d.get('user_id', 0))

        if not crop:
            return jsonify({"error": "Crop name is required"}), 400

        # ── ML Prediction ───────────────────────────────────────────────
        features = np.array([[N, P, K, ph]])
        predicted_crop = MODEL.predict(features)[0]

        # ── Compute fertilizer recommendation ──────────────────────────
        result = compute_recommendation(crop, N, P, K, ph, soil_type)
        result['predicted_crop'] = predicted_crop

        # ── Save record to DB ─────────────────────────────────────────
        conn = get_db()
        conn.execute(
            "INSERT INTO records(user_id,crop,nitrogen,phosphorus,potassium,ph,soil_type,moisture,location,result,predicted_crop) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (user_id, crop, N, P, K, ph, soil_type, moisture, location, json.dumps(result), predicted_crop)
        )
        conn.commit()
        conn.close()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/history/<int:uid>')
def history(uid):
    conn = get_db()
    rows = conn.execute(
        "SELECT id,crop,nitrogen,phosphorus,potassium,ph,soil_type,predicted_crop,created_at FROM records WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
        (uid,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])
if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 10000))
    print("\n" + "="*50)
    print("  ✅ AgriSense Backend Running!")
    print("  🌐 Server starting...")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=port)
