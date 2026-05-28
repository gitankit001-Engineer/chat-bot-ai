import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("custom_job_dataset.csv")

df = df.dropna(subset=['Required_Skills'])

df['Required_Skills'] = (
    df['Required_Skills']
    .astype(str)
    .str.lower()
)

# ============================================================
# TRENDING SKILLS
# ============================================================

def get_trending_skills():

    all_skills = " ".join(df['Required_Skills']).split()

    common_skills = Counter(all_skills).most_common(3)

    trending = " ".join([
        skill for skill, count in common_skills
    ])

    return trending

# ============================================================
# MAIN RECOMMENDATION FUNCTION
# ============================================================

def get_recommendations(user_input):

    if not user_input.strip():

        user_input = get_trending_skills()

    user_profile = user_input.lower()

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )

    all_text_data = df['Required_Skills'].tolist()

    all_text_data.append(user_profile)

    tfidf_matrix = vectorizer.fit_transform(all_text_data)

    item_vectors = tfidf_matrix[:-1]

    user_vector = tfidf_matrix[-1]

    similarity_scores = cosine_similarity(
        user_vector,
        item_vectors
    ).flatten()

    df['Match_Score'] = similarity_scores

    filtered_df = df[df['Match_Score'] > 0.1]

    sorted_df = filtered_df.sort_values(
        by='Match_Score',
        ascending=False
    )
    unique_df = sorted_df.drop_duplicates(subset=['Job_Role'], keep='first')
    top_matches = unique_df.head(5)

    results = []

    for _, row in top_matches.iterrows():

        score = round(row['Match_Score'] * 100, 2)

        user_skills = set(user_profile.split())

        job_skills = set(
            row['Required_Skills'].split()
        )

        matched_skills = list(
            user_skills.intersection(job_skills)
        )

        missing_skills = list(
            job_skills.difference(user_skills)
        )[:5]

        if row['Match_Score'] > 0.7:
            confidence = "Excellent Match"

        elif row['Match_Score'] > 0.4:
            confidence = "Moderate Match"

        else:
            confidence = "Weak Match"

        results.append({

            "role": row['Job_Role'],

            "score": score,

            "confidence": confidence,

            "matched_skills": matched_skills,

            "missing_skills": missing_skills,

            "full_stack": row['Required_Skills']

        })

    return results
# ============================================================
# EXECUTION BLOCK (Yahan se code actually run hoga)
# ============================================================

if __name__ == "__main__":
    # Apni pasand ki koi bhi skills daal kar check kar
    test_skills = "python machine learning power bi"
    
    print(f"🔍 Searching jobs for: {test_skills}\n")
    
    # Function ko call karo aur result variable mein store karo
    results = get_recommendations(test_skills)
    
    # Results ko line-by-line print karo
    for i, job in enumerate(results, 1):
        print(f"Rank #{i} | Role: {job['role']}")
        print(f"Match Score: {job['score']}% ({job['confidence']})")
        print(f"Matched Skills: {job['matched_skills']}")
        print("-" * 40)


print("\n> System Shutting Down...")
print("> Thank you for using DecodeLabs Matchmaker!\n")
