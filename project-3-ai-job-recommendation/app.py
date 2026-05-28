from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# --- Load and Clean Data ---
try:
    df = pd.read_csv('custom_job_dataset.csv')
    df.dropna(subset=['Job_Requirements'], inplace=True)
    df.drop_duplicates(subset=['Job_ID'], inplace=True)
    df['Job_Requirements'] = df['Job_Requirements'].astype(str).str.lower()
    
    # Extract unique skills for the Dropdown menu (Assuming comma-separated skills)
    all_skills_set = set()
    for req in df['Job_Requirements']:
        # comma se split karke extra space hata do
        skills = [s.strip().title() for s in req.split(',') if s.strip()]
        all_skills_set.update(skills)
    unique_skills = sorted(list(all_skills_set))
    
except FileNotFoundError:
    print("⚠️ ERROR: 'Job Dataset.csv' not found.")
    df = pd.DataFrame()
    unique_skills = ["Python", "Java", "Sql", "Aws", "Docker"] # Fallback

@app.route('/')
def home():
    # Render template aur dynamically skills pass karna dropdown ke liye
    return render_template('index.html', skills=unique_skills)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    skills = data.get('skills', ["", "", ""])
    experience = data.get('experience', "Beginner") # Captured experience
    
    s1, s2, s3 = (skills + ["", "", ""])[:3]
    
    if not s1 and not s2 and not s3:
        user_profile = "python sql javascript"
    else:
        user_profile = f"{s1} {s2} {s3}".lower()
        
    vectorizer = TfidfVectorizer()
    all_text = df['Job_Requirements'].tolist()
    all_text.append(user_profile)
    
    tfidf_matrix = vectorizer.fit_transform(all_text)
    item_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]
    
    scores = cosine_similarity(user_vector, item_vectors).flatten()
    
    df_copy = df.copy()
    df_copy['Match_Score'] = scores
    
    # Experience based logic can be added here in future. For now, we sort by Match.
    top_3 = df_copy.sort_values(by='Match_Score', ascending=False).head(3)
    
    results = []
    for _, row in top_3.iterrows():
        results.append({
            "job_id": str(row['Job_ID']),
            "match": round(row['Match_Score'] * 100, 2),
            "stack": str(row['Job_Requirements']).title()
        })
        
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
