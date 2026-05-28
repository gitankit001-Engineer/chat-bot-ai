# ===========================================
# APP.PY — AI Tech Recommender (ngrok + UI)
# ===========================================

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template_string
from pyngrok import ngrok

app = Flask(__name__)

# --- Load and Clean Data ---
try:
    # Use your generated dataset
    df = pd.read_csv('custom_job_dataset.csv')
    df.dropna(subset=['Required_Skills'], inplace=True)
    df['Required_Skills'] = df['Required_Skills'].astype(str).str.lower()
    
    # Extract unique skills for the HTML Dropdown
    all_skills_set = set()
    for req in df['Required_Skills']:
        skills = [s.strip().title() for s in req.split(',') if s.strip()]
        all_skills_set.update(skills)
    unique_skills = sorted(list(all_skills_set))
    
except FileNotFoundError:
    print("⚠️ ERROR: 'custom_job_dataset.csv' not found. Using fallback data.")
    df = pd.DataFrame()
    unique_skills = ["Python", "Sql", "Javascript", "Aws", "Docker"]

# --- Integrated HTML + CSS + JS Template ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DecodeLabs | Tech Recommender</title>
    <style>
        :root {
            --bg-dark: #0b1320;
            --glass-bg: rgba(20, 30, 48, 0.6);
            --glass-border: rgba(255, 255, 255, 0.08);
            --accent: #00e5ff;
            --accent-hover: #00b8cc;
            --text-main: #ffffff;
            --text-muted: #8a9bb3;
            --card-bg: rgba(0, 0, 0, 0.3);
        }

        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at top left, #14283b, var(--bg-dark));
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }

        .main-wrapper {
            display: flex;
            gap: 30px;
            width: 100%;
            max-width: 1200px;
        }

        /* --- LEFT PANEL (Controls) --- */
        .left-panel {
            flex: 1;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 40px;
            border-radius: 24px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 30px 60px rgba(0,0,0,0.4);
            height: fit-content;
        }

        .left-panel h1 {
            font-size: 32px;
            font-weight: 800;
            margin: 0 0 5px 0;
            color: var(--text-main);
            letter-spacing: -0.5px;
        }

        .left-panel p.subtitle {
            color: var(--accent);
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin: 0 0 35px 0;
        }

        .form-group { margin-bottom: 22px; }
        
        /* Professional Label Styling */
        label { 
            display: block; 
            margin-bottom: 8px; 
            font-size: 12px; 
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted); 
        }
        
        select {
            width: 100%; 
            padding: 14px 16px; 
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--glass-border); 
            border-radius: 12px;
            color: white; 
            font-size: 15px; 
            font-family: 'Inter', sans-serif;
            outline: none; 
            appearance: none;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.15);
        }
        
        select option { background-color: #0b1320; color: white; }

        .primary-btn {
            width: 100%; 
            padding: 18px; 
            margin-top: 15px;
            background: var(--accent); 
            color: #000; 
            border: none;
            border-radius: 12px; 
            font-size: 16px; 
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            cursor: pointer; 
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px rgba(0, 229, 255, 0.2);
        }
        
        .primary-btn:hover { 
            background: var(--accent-hover); 
            transform: translateY(-2px);
            box-shadow: 0 15px 25px rgba(0, 229, 255, 0.3);
        }

        /* --- RIGHT PANEL (Results) --- */
        .right-panel {
            flex: 1.2;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 40px;
            border-radius: 24px;
            border: 1px solid var(--glass-border);
            max-height: 85vh; 
            overflow-y: auto;  
            box-shadow: 0 30px 60px rgba(0,0,0,0.4);
        }

        .right-panel h2 {
            font-size: 22px;
            font-weight: 600;
            color: var(--text-main);
            margin: 0 0 25px 0;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--glass-border);
        }

        /* Custom Scrollbar */
        .right-panel::-webkit-scrollbar { width: 6px; }
        .right-panel::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); border-radius: 10px; }
        .right-panel::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 10px; }
        .right-panel::-webkit-scrollbar-thumb:hover { background: var(--accent); }

        .loader { 
            display: none; 
            text-align: center; 
            margin-top: 25px; 
            color: var(--accent); 
            font-weight: 600;
            font-size: 14px;
            letter-spacing: 1px;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        /* --- JOB RESULT CARDS --- */
        .result-card {
            background: var(--card-bg); 
            border-left: 4px solid var(--accent);
            border-radius: 12px; 
            padding: 25px; 
            margin-bottom: 20px;
            transition: all 0.3s ease;
            border-top: 1px solid rgba(255,255,255,0.03);
            border-right: 1px solid rgba(255,255,255,0.03);
            border-bottom: 1px solid rgba(255,255,255,0.03);
        }
        
        .result-card:hover {
            transform: translateX(5px);
            background: rgba(255, 255, 255, 0.05);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            border-left-color: #fff;
        }

        .result-card h3 { 
            margin: 0 0 10px 0; 
            font-size: 18px; 
            font-weight: 600;
            color: var(--text-main);
        }
        
        .match-stat { 
            font-size: 32px; 
            font-weight: 800; 
            color: var(--accent); 
            float: right; 
            margin-top: -35px;
            text-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
        }
        
        .stack-list { 
            margin: 15px 0 0 0; 
            color: var(--text-muted); 
            font-size: 14px; 
            line-height: 1.6;
        }

        .stack-list strong {
            color: #fff;
            font-weight: 600;
        }

        /* Responsive Design */
        @media (max-width: 900px) {
            .main-wrapper { flex-direction: column; }
            .right-panel { max-height: none; overflow-y: visible; }
        }
    </style>
</head>
<body>
<div class="main-wrapper">
    <div class="left-panel">
        <h1>Digital Matchmaker</h1>
        <p>AI-Powered Career Profiling System</p>
    
        <div class="form-group">
            <label>Experience Level</label>
            <select id="experience">
                <option value="Beginner">Beginner (0-1 Yrs)</option>
                <option value="Intermediate">Intermediate (1-3 Yrs)</option>
                <option value="Advanced">Advanced (3-5+ Yrs)</option>
            </select>
        </div>

        <div class="form-group">
            <label>Core Skill 1</label>
            <select id="s1">
                <option value="">-- Select Skill --</option>
                {% for skill in skills %}
                <option value="{{ skill }}">{{ skill }}</option>
                {% endfor %}
            </select>
        </div>
        
        <div class="form-group">
            <label>Core Skill 2</label>
            <select id="s2">
                <option value="">-- Select Skill --</option>
                {% for skill in skills %}
                <option value="{{ skill }}">{{ skill }}</option>
                {% endfor %}
            </select>
        </div>
        
        <div class="form-group">
            <label>Core Skill 3</label>
            <select id="s3">
                <option value="">-- Select Skill --</option>
                {% for skill in skills %}
                <option value="{{ skill }}">{{ skill }}</option>
                {% endfor %}
            </select>
        </div>

        <div class="form-group">
            <label>Core Skill 4</label>
            <select id="s4">
                <option value="">-- Select Skill --</option>
                {% for skill in skills %}
                <option value="{{ skill }}">{{ skill }}</option>
                {% endfor %}
            </select>
        </div>

        <div class="form-group">
            <label>Core Skill 5</label>
            <select id="s5">
                <option value="">-- Select Skill --</option>
                {% for skill in skills %}
                <option value="{{ skill }}">{{ skill }}</option>
                {% endfor %}
            </select>
        </div>
        
        <button class="primary-btn" onclick="getRecommendations()">Initialize AI Engine</button>
        
        <div class="loader" id="loader">Running AI Math...</div>
        
    </div> <div class="right-panel">
        <h2>Recommended Roles</h2>
        <div id="results"></div>
    </div> </div>

<script>
    async function getRecommendations() {
        const s1 = document.getElementById('s1').value;
        const s2 = document.getElementById('s2').value;
        const s3 = document.getElementById('s3').value;
        const s4 = document.getElementById('s4').value;
        const s5 = document.getElementById('s5').value;

        document.getElementById('loader').style.display = 'block';
        document.getElementById('results').innerHTML = '';

        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ skills: [s1, s2, s3, s4, s5] })
            });
            
            const data = await response.json();
            document.getElementById('loader').style.display = 'none';
            
            let html = '';
            data.forEach((item, index) => {
                html += `
                <div class="result-card">
                    <h3>Rank #${index + 1} | ${item.role}</h3>
                    <div class="match-stat">${item.match}%</div>
                    <p class="stack-list"><strong>Required Stack:</strong><br>${item.stack}</p>
                </div>`;
            });
            document.getElementById('results').innerHTML = html;
        } catch (error) {
            document.getElementById('loader').style.display = 'none';
            document.getElementById('results').innerHTML = `<p style="color:red;">Error connecting to engine.</p>`;
        }
    }
</script>

</body>
</html>
"""

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_PAGE, skills=unique_skills)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    skills = data.get('skills', ["", "", "", "", ""])
    
    s1, s2, s3, s4, s5 = (skills + ["", "", "", "", ""])[:5]  # Ensure we have 5 skills, fill missing with empty strings
    
    # Cold start baseline
    if not s1 and not s2 and not s3:
        user_profile = "python sql javascript"
    else:
        user_profile = f"{s1} {s2} {s3} {s4} {s5}".lower()
        
    vectorizer = TfidfVectorizer()
    all_text = df['Required_Skills'].tolist()
    all_text.append(user_profile)
    
    tfidf_matrix = vectorizer.fit_transform(all_text)
    item_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]
    
    scores = cosine_similarity(user_vector, item_vectors).flatten()
    
    df_copy = df.copy()
    df_copy['Match_Score'] = scores
    
    top_10 = df_copy.sort_values(by='Match_Score', ascending=False).head(10)
    
    results = []
    for _, row in top_10.iterrows():
        results.append({
            "role": str(row.get('Job_Role', 'Unknown Role')),
            "match": round(row['Match_Score'] * 100, 2),
            "stack": str(row['Required_Skills']).title()
        })
        
    return jsonify(results)

# --- Run Flask + ngrok ---
if __name__ == '__main__':
    # Tera Auth token set kiya hai
    ngrok.set_auth_token("34vanTj6wOahEuYzqikHCgVsvUh_55byjA96RB6ZdYoGDF7Cy")
    
    print("🚀 Starting Flask Engine...")
    public_url = ngrok.connect(5000).public_url
    print(f"\n🌍 PUBLIC URL ACTIVE: {public_url}\n")
    
    app.run(port=5000)     
