# =======================================================
# APP.PY — AI Tech Recommender (Hybrid Match + ngrok)
# =======================================================

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template_string
from pyngrok import ngrok

app = Flask(__name__)

# --- Load and Clean Data ---
try:
    # Tumhari generated file ko read karna
    df = pd.read_csv('custom_job_dataset.csv')
    df.dropna(subset=['Required_Skills'], inplace=True)
    df['Required_Skills'] = df['Required_Skills'].astype(str).str.lower()
    
    # Dropdown ke liye saari unique skills nikalna
    all_skills_set = set()
    for req in df['Required_Skills']:
        skills = [s.strip().title() for s in req.split(',') if s.strip()]
        all_skills_set.update(skills)
    unique_skills = sorted(list(all_skills_set))
    
except FileNotFoundError:
    print("⚠️ ERROR: 'custom_job_dataset.csv' not found. Using fallback data.")
    df = pd.DataFrame()
    unique_skills = ["Python", "Sql", "Machine Learning", "Linux", "Power Bi", "Cyber Security"]

# --- Integrated HTML + CSS + JS Template ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DecodeLabs | Tech Recommender</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
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
        }

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
            margin: 0 0 25px 0;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--glass-border);
        }

        .right-panel::-webkit-scrollbar { width: 6px; }
        .right-panel::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 10px; }
        .right-panel::-webkit-scrollbar-thumb:hover { background: var(--accent); }

        .loader { 
            display: none; 
            text-align: center; 
            margin-top: 25px; 
            color: var(--accent); 
            font-weight: 600;
        }
        
        .result-card {
            background: var(--card-bg); 
            border-left: 4px solid var(--accent);
            border-radius: 12px; 
            padding: 25px; 
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .result-card:hover {
            transform: translateX(5px);
            background: rgba(255, 255, 255, 0.05);
        }

        .result-card h3 { margin: 0 0 10px 0; font-size: 18px; font-weight: 600;}
        .match-stat { 
            font-size: 32px; 
            font-weight: 800; 
            color: var(--accent); 
            float: right; 
            margin-top: -35px;
            text-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
        }
        .stack-list { margin: 15px 0 0 0; color: var(--text-muted); font-size: 14px; line-height: 1.6; }
        .stack-list strong { color: #fff; }

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
        <p class="subtitle">AI-Powered Career Profiling</p>
    
        <div class="form-group">
            <label>Experience Level</label>
            <select id="experience">
                <option value="Beginner">Beginner (0-1 Yrs)</option>
                <option value="Intermediate">Intermediate (1-3 Yrs)</option>
                <option value="Advanced">Advanced (3-5+ Yrs)</option>
            </select>
        </div>

        {% for i in range(1, 6) %}
        <div class="form-group">
            <label>Core Skill {{ i }}</label>
            <select id="s{{ i }}">
                <option value="">-- Select Skill --</option>
                {% for skill in skills %}
                <option value="{{ skill }}">{{ skill }}</option>
                {% endfor %}
            </select>
        </div>
        {% endfor %}
        
        <button class="primary-btn" onclick="getRecommendations()">Initialize AI Engine</button>
        <div class="loader" id="loader">Processing Vector Math...</div>
    </div> 
    
    <div class="right-panel">
        <h2>Recommended Roles (Top 5)</h2>
        <div id="results">
            <div style="text-align: center; color: var(--text-muted); margin-top: 50px; font-size: 14px;">
                Select your skills and initialize the engine to see your matches here.
            </div>
        </div>
    </div> 
</div>

<script>
    async function getRecommendations() {
        const skills = [
            document.getElementById('s1').value,
            document.getElementById('s2').value,
            document.getElementById('s3').value,
            document.getElementById('s4').value,
            document.getElementById('s5').value
        ];

        document.getElementById('loader').style.display = 'block';
        document.getElementById('results').innerHTML = '';

        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ skills: skills })
            });
            
            const data = await response.json();
            document.getElementById('loader').style.display = 'none';
            
            let html = '';
            if(data.length === 0) {
                html = '<p style="text-align:center; color:var(--text-muted);">No exact matches found. Try changing skills.</p>';
            } else {
                data.forEach((item, index) => {
                    html += `
                    <div class="result-card">
                        <div class="match-stat">${item.match}%</div>
                        <h3>Rank #${index + 1} | ${item.role}</h3>
                        <p class="stack-list"><strong>Required Stack:</strong><br>${item.stack}</p>
                    </div>`;
                });
            }
            document.getElementById('results').innerHTML = html;
        } catch (error) {
            document.getElementById('loader').style.display = 'none';
            document.getElementById('results').innerHTML = `
                <div class="result-card" style="border-left-color: #ff3366;">
                    <h3 style="color: #ff3366;">Connection Error</h3>
                    <p class="stack-list">Failed to fetch data from the AI engine.</p>
                </div>`;
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
    
    # Filter empty strings
    valid_skills = [s.strip().lower() for s in skills if s.strip()]
    
    if not valid_skills:
        return jsonify([])
        
    user_profile = " ".join(valid_skills)
        
    # --- PHASE 1: AI / ML SCORE (Cosine Similarity) ---
    vectorizer = TfidfVectorizer()
    all_text = df['Required_Skills'].tolist()
    all_text.append(user_profile)
    
    tfidf_matrix = vectorizer.fit_transform(all_text)
    item_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]
    
    ai_scores = cosine_similarity(user_vector, item_vectors).flatten()
    
    # --- PHASE 2: EXACT MATCH INTERSECTION SCORE ---
    exact_match_scores = []
    for req in df['Required_Skills']:
        req_skills_list = [s.strip().lower() for s in req.split(',')]
        
        # Calculate intersection
        match_count = sum(1 for skill in valid_skills if skill in req_skills_list)
        score = match_count / len(valid_skills)
        exact_match_scores.append(score)
        
    # --- PHASE 3: HYBRID SORTING ---
   # --- PHASE 3: HYBRID SORTING & DIVERSITY FILTER ---
    df_copy = df.copy()
    hybrid_scores = [ (ai * 0.4) + (exact * 0.6) for ai, exact in zip(ai_scores, exact_match_scores) ]
    df_copy['Match_Score'] = hybrid_scores
    
    # 1. Sabse pehle Match Score ke hisaab se sort karo (Highest to Lowest)
    df_sorted = df_copy.sort_values(by='Match_Score', ascending=False)
    
    # 2. DIVERSITY FILTER: Ek job role ko sirf ek baar rakho (Duplicates hata do)
    # Yeh line ensure karegi ki Data Scientist sirf ek baar aaye aur baaki jagah doosre roles lein
    if 'Job_Role' in df_sorted.columns:
        df_unique = df_sorted.drop_duplicates(subset=['Job_Role'], keep='first')
    else:
        df_unique = df_sorted # Fallback agar column name alag ho
    
    # 3. Ab in unique roles mein se Top 10 nikal lo
    top_5 = df_unique.head(5)
    
    results = []
    for _, row in top_5.iterrows():
        # Match score agar 0 ho toh usko hide kar do
        match_percentage = round(row['Match_Score'] * 100, 2)
        if match_percentage > 0:
            results.append({
                "role": str(row.get('Job_Role', 'Technical Specialist')).title(),
                "match": match_percentage,
                "stack": str(row.get('Required_Skills', '')).title()
            })
        
    return jsonify(results)

if __name__ == '__main__':
    ngrok.set_auth_token("34vanTj6wOahEuYzqikHCgVsvUh_55byjA96RB6ZdYoGDF7Cy")
    print("🚀 Starting Flask Engine...")
    public_url = ngrok.connect(5000).public_url
    print(f"\n🌍 PUBLIC URL ACTIVE: {public_url}\n")
    app.run(port=5000)
