# 🌱 AgriSense — Fertilizer Recommendation System

## ▶️ HOW TO RUN (Important — read this!)

### Step 1: Install dependencies
Open terminal in VS Code, navigate to the backend folder:
```
cd backend
pip install flask flask-cors
```

### Step 2: Start the backend
```
python app.py
```
You will see:
```
✅ AgriSense Backend Running!
🌐 Open: http://localhost:5000
```

### Step 3: Open the app
Open your browser and go to:
```
http://localhost:5000
```

⚠️ IMPORTANT: Always use http://localhost:5000 — do NOT open index.html by double-clicking it.
The frontend must be served by Flask to avoid browser security (CORS) errors.

---

## 📁 Project Structure
```
fertilizer-system/
├── backend/
│   ├── app.py           ← Flask backend (run this)
│   ├── requirements.txt ← pip dependencies
│   └── fertilizer.db    ← SQLite database (auto-created)
└── frontend/
    ├── index.html       ← Login / Register
    └── dashboard.html   ← Main app
```

## ✅ Features
- Register & Login with password hashing
- Type any crop name manually
- NPK soil analysis with Low/Medium/High status
- Fertilizer recommendations (Urea, DAP, MOP)
- Cost savings calculation
- Organic alternatives
- Farming tips
- Record history per user
- NPK reference guide
