import pandas as pd
import random

# ============================================================
# ADVANCED AI JOB DATASET GENERATOR
# ============================================================

roles_and_skills = {

    "Data Scientist": [
        "python", "sql", "machine learning", "pandas",
        "numpy", "tensorflow", "deep learning",
        "nlp", "statistics", "scikit-learn",
        "matplotlib", "seaborn", "feature engineering"
    ],

    "Machine Learning Engineer": [
        "python", "tensorflow", "pytorch",
        "mlops", "docker", "kubernetes",
        "aws", "deep learning", "model deployment",
        "fastapi", "flask", "computer vision"
    ],

    "AI Engineer": [
        "python", "llms", "transformers",
        "langchain", "openai api", "prompt engineering",
        "tensorflow", "huggingface", "vector databases",
        "rag", "deep learning", "nlp"
    ],

    "Backend Developer": [
        "python", "django", "flask",
        "nodejs", "express", "java",
        "spring boot", "mongodb", "postgresql",
        "rest api", "microservices", "redis"
    ],

    "Frontend Developer": [
        "html", "css", "javascript",
        "react", "nextjs", "vue",
        "angular", "tailwind", "bootstrap",
        "redux", "figma", "responsive design"
    ],

    "Full Stack Developer": [
        "javascript", "react", "nodejs",
        "mongodb", "express", "python",
        "django", "sql", "rest api",
        "tailwind", "docker", "firebase"
    ],

    "Cloud Architect": [
        "aws", "azure", "gcp",
        "terraform", "docker", "kubernetes",
        "linux", "networking", "ci/cd",
        "cloud security", "microservices"
    ],

    "DevOps Engineer": [
        "docker", "kubernetes", "jenkins",
        "github actions", "ansible",
        "terraform", "linux", "aws",
        "bash", "monitoring", "prometheus"
    ],

    "Cybersecurity Analyst": [
        "ethical hacking", "penetration testing",
        "siem", "firewalls", "linux",
        "network security", "cryptography",
        "soc", "vulnerability assessment",
        "wireshark", "incident response"
    ],

    "Blockchain Developer": [
        "solidity", "ethereum", "smart contracts",
        "web3", "hardhat", "rust",
        "cryptography", "defi", "polygon",
        "blockchain security"
    ],

    "Mobile App Developer": [
        "flutter", "dart", "firebase",
        "android", "ios", "swift",
        "kotlin", "react native",
        "api integration", "mobile ui"
    ],

    "Android Developer": [
        "kotlin", "java", "android studio",
        "firebase", "jetpack compose",
        "rest api", "sqlite", "material design"
    ],

    "iOS Developer": [
        "swift", "xcode", "ios",
        "swiftui", "firebase",
        "api integration", "core data"
    ],

    "Game Developer": [
        "unity", "unreal engine",
        "c#", "c++", "game physics",
        "3d graphics", "blender",
        "multiplayer networking"
    ],

    "Data Analyst": [
        "excel", "sql", "power bi",
        "tableau", "python", "pandas",
        "statistics", "data visualization",
        "dashboarding"
    ],

    "Business Analyst": [
        "excel", "sql", "power bi",
        "requirement gathering", "documentation",
        "data analysis", "jira", "communication"
    ],

    "UI/UX Designer": [
        "figma", "adobe xd", "wireframing",
        "prototyping", "user research",
        "design systems", "responsive design"
    ],

    "Database Administrator": [
        "mysql", "postgresql", "oracle",
        "database tuning", "backup recovery",
        "sql", "performance optimization"
    ],

    "Network Engineer": [
        "routing", "switching", "cisco",
        "network security", "vpn",
        "linux", "tcp/ip", "firewalls"
    ],

    "Site Reliability Engineer": [
        "kubernetes", "docker", "linux",
        "aws", "monitoring", "terraform",
        "incident management", "prometheus"
    ],

    "Software Tester": [
        "manual testing", "selenium",
        "automation testing", "jira",
        "test cases", "bug tracking",
        "api testing"
    ],

    "QA Engineer": [
        "selenium", "cypress",
        "automation testing", "api testing",
        "postman", "performance testing"
    ],

    "Computer Vision Engineer": [
        "opencv", "python", "tensorflow",
        "image processing", "deep learning",
        "cnn", "yolo", "object detection"
    ],

    "NLP Engineer": [
        "transformers", "bert",
        "huggingface", "nlp",
        "python", "text classification",
        "sentiment analysis", "llms"
    ],

    "AR/VR Developer": [
        "unity", "unreal engine",
        "arcore", "arkit",
        "3d modeling", "c#", "vr interaction"
    ],

    "Embedded Systems Engineer": [
        "c", "c++", "microcontrollers",
        "arduino", "raspberry pi",
        "iot", "embedded linux"
    ]
}

dataset = []

total_rows_to_generate = 10000

print(f"> Generating {total_rows_to_generate} rows of ADVANCED IT dataset...")

for _ in range(total_rows_to_generate):

    role = random.choice(list(roles_and_skills.keys()))

    possible_skills = roles_and_skills[role]

    min_skills = min(4, len(possible_skills))
    max_skills = min(8, len(possible_skills))

    num_skills = random.randint(min_skills, max_skills)

    selected_skills = random.sample(possible_skills, num_skills)

    skills_string = ", ".join(selected_skills)

    dataset.append({
        "Job_Role": role,
        "Required_Skills": skills_string
    })

# Create DataFrame
df = pd.DataFrame(dataset)

# Remove duplicates
df = df.drop_duplicates()

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save CSV
df.to_csv("custom_job_dataset.csv", index=False)

print("\n> SUCCESS: Advanced dataset generated!")
print(f"> Total Rows: {len(df)}")
print(f"> Total Unique Roles: {df['Job_Role'].nunique()}")

print("\n> Dataset Ready for AI Recommendation System.")
print(df['Job_Role'].value_counts())