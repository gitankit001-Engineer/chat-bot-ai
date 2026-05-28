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

    top_matches = sorted_df.head(3)

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


# # ============================================================
# # PROJECT 3 — AI Tech Stack Recommender (Digital Matchmaker)
# # Pipeline: Ingestion -> Scoring -> Sorting -> Filtering
# # ============================================================
# import pandas as pd
# from collections import Counter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # --- STEP 1: INGESTION & DATA CLEANING ---
# try:
#     df = pd.read_csv('custom_job_dataset.csv')
#     df = df.dropna(subset=['Required_Skills'])
#     df['Required_Skills'] = df['Required_Skills'].astype(str).str.lower()
    
#     print(f"> Dataset Ready! Total Unique Jobs available: {len(df)}")
    
# except FileNotFoundError:
#     print("⚠️ ERROR: 'custom_job_dataset.csv' not found. Please ensure the file is in the folder.")
#     exit()
# def get_trending_skills():
#     all_skills = " ".join(df['Required_Skills']).split()
#     common_skills = Counter(all_skills).most_common(3)

#     trending = " ".join([skill for skill, count in common_skills])

#     return trending


# def get_user_profile():

#     print("\n" + "="*60)
#     print(" 🚀 DECODELABS DIGITAL MATCHMAKER - SYSTEM ONLINE")
#     print("="*60)

#     print("> Enter your profile information:\n")

#     skill_1 = input("Skill 1: ").strip()
#     skill_2 = input("Skill 2: ").strip()
#     skill_3 = input("Skill 3: ").strip()

#     experience = input("Experience Level (Fresher/Intermediate/Advanced): ").strip()

#     # domain = input("Preferred Domain (AI/Web/Data/Cloud): ").strip()

#     # Dynamic Cold Start Handling
#     if not skill_1 and not skill_2 and not skill_3:

#         trending_stack = get_trending_skills()

#         print("\n[!] COLD START DETECTED")
#         print(f"> Using Trending Skills: {trending_stack}")

#         return trending_stack.lower()

#     user_profile = f"""
#     {skill_1}
#     {skill_2}
#     {skill_3}
#     {experience}

#     """

#     return user_profile.lower()
# # def get_user_profile():
# #     print("\n" + "="*60)
# #     print(" 🚀 DECODELABS DIGITAL MATCHMAKER - SYSTEM ONLINE")
# #     print("="*60)
# #     print("> Enter your top 3 skills (e.g., Python, AWS, Docker):")
    
# #     skill_1 = input("Skill 1: ").strip()
# #     skill_2 = input("Skill 2: ").strip()
# #     skill_3 = input("Skill 3: ").strip()
    
# #     # COLD START BYPASS: If the user enters nothing, default to a baseline [cite: 261-262, 266]
# #     if not skill_1 and not skill_2 and not skill_3:
# #         print("\n[!] COLD START DETECTED: Defaulting to Trending Tech Stack...")
# #         return "python sql javascript"
    
# #     return f"{skill_1} {skill_2} {skill_3}".lower()

# # --- EXECUTE PIPELINE ---
# if __name__ == "__main__":
    
#     # 1. INGESTION: Get user profile [cite: 200, 204]
#     user_profile = get_user_profile()
#     print(f"\n> Analyzed User Vector: [{user_profile}]")
    
#     # --- STEP 2: SCORING (Vector Mapping & Cosine Similarity) ---
#     print("> Initializing TF-IDF Vector Space...")
    
#     # Initialize the TF-IDF Vectorizer [cite: 118-120]
#     vectorizer = TfidfVectorizer(
#     stop_words='english',
#     ngram_range=(1,2),
#     max_features=5000
# )
    
#     # Combine item dataset and user profile into one list (Shared Vocabulary) [cite: 90-94]
#     all_text_data = df['Required_Skills'].tolist()
#     all_text_data.append(user_profile)
    
#     # Transform all text into numerical vectors [cite: 90-91]
#     tfidf_matrix = vectorizer.fit_transform(all_text_data)
    
#     # Separate the item vectors and the user vector
#     item_vectors = tfidf_matrix[:-1] 
#     user_vector = tfidf_matrix[-1]   
    
#     # Calculate Cosine Similarity (measuring the angle between vectors) [cite: 163-165, 304]
#     print("> Calculating Cosine Angular Alignment...")
#     similarity_scores = cosine_similarity(user_vector, item_vectors).flatten()
    
#     # Add scores to our dataframe
#     df['Match_Score'] = similarity_scores
#     # Remove weak recommendations
#     df = df[df['Match_Score'] > 0.1]
    
#     # --- STEP 3: SORTING ---
#     print("> Sorting results by mathematical relevance...")
#     # Sort the dataframe by Match_Score in descending order [cite: 225-228]
#     df_sorted = df.sort_values(by='Match_Score', ascending=False)
    
#     # --- STEP 4: FILTERING (Top-N Output) ---
#     # Truncate to prevent choice overload. Show only the Top 3 matches [cite: 238-239, 305]
#     top_3_matches = df_sorted.head(3)
#     if top_3_matches.empty:
#         print("\n⚠️ No strong matches found.")
#         print("> Try adding more technical skills.")
#         exit()
    
#     print("\n" + "="*60)
#     print(" 🎯 FINAL RECOMMENDATIONS (TOP-3 MATCHES)")
#     print("="*60)
    
#     # Display the results professionally
#     rank = 1
#     for index, row in top_3_matches.iterrows():

#         match_percentage = round(row['Match_Score'] * 100, 2)

#         user_skills = set(user_profile.split())
#         job_skills = set(row['Required_Skills'].split())

#         matched_skills = user_skills.intersection(job_skills)

#         missing_skills = job_skills.difference(user_skills)

#         # Recommendation Confidence
#         if row['Match_Score'] > 0.7:
#             confidence = "Excellent Match"

#         elif row['Match_Score'] > 0.4:
#             confidence = "Moderate Match"

#         else:
#             confidence = "Weak Match"

#         print(f"[{rank}] Role : {row['Job_Role']}")
#         print(f"    Match Score : {match_percentage}%")
#         print(f"    Confidence  : {confidence}")

#         print(f"    Matched Skills : {', '.join(matched_skills)}")

#         print(f"    Missing Skills : {', '.join(list(missing_skills)[:5])}")

#         print(f"    Full Stack : {row['Required_Skills'].title()}\n")

#         rank += 1
        
#     print("="*60)
#     print(" Recommendation Pipeline Completed Successfully ")
#     print("="*60)

#     print("\n> System Shutting Down...")
#     print("> Thank you for using DecodeLabs Matchmaker!\n")