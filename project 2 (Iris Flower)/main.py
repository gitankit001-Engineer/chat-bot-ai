# ============================================================
# PROJECT 2 — Iris KNN Classifier
# model.py | Section 1: Data Load + Explore
# ============================================================

# --- 1. Libraries import karo ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# --- 2. Dataset load karo (koi CSV nahi chahiye!) ---
iris = load_iris()

# --- 3. Features (X) aur Labels (y) alag karo ---
X = iris.data # Shape: (150, 4) → 150 phool, 4 features
y = iris.target # Shape: (150,) → 0=Setosa, 1=Versicolor, 2=Virginica
feature_names = iris.feature_names
# --- 4. Readable DataFrame banao (explore karne ke liye) ---
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# --- 5. Data explore karo ---
print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Total samples : {X.shape[0]}")
print(f"Total features: {X.shape[1]}")
print(f"Classes : {list(iris.target_names)}")
print()
print("--- First 5 rows ---")
print(df.head())
print()
print("--- Class distribution (balanced check) ---")
print(df['species'].value_counts())
print()
print("--- Basic stats ---")
print(df.describe().round(2))

# ============================================================
# Section 2: Preprocessing — Scaling + Train/Test Split
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Step 1: Train/Test Split (80% train, 20% test) ---
# shuffle=True → data randomize hoga (order bias hatega)
# random_state=42 → har baar same split milegi (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    stratify=y,
    random_state=42
)

# --- Step 2: Scaler banao aur SIRF train data par fit karo ---
# WARNING: scaler ko test data par fit mat karna!
# Warna "data leakage" ho jayegi (machine future dekh legi)
scaler = StandardScaler()
scaler.fit(X_train) # mean aur std sirf training data se seekhe

# --- Step 3: Dono sets ko transform karo ---
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 4: Scaler ke mean/std print karo (JS ke liye zaroori!) ---
print("="*50)
print("PREPROCESSING COMPLETE")
print("="*50)
print(f"Training samples : {X_train_scaled.shape[0]}")
print(f"Testing samples : {X_test_scaled.shape[0]}")
print()
print("--- Scaler values (JS website ke liye save karo!) ---")
print(f"Mean : {np.round(scaler.mean_, 4)}")
print(f"Std : {np.round(scaler.scale_, 4)}")
print()
print("--- Scaling check (mean ~0, std ~1 hona chahiye) ---")
print(f"Scaled mean : {np.round(X_train_scaled.mean(axis=0), 2)}")
print(f"Scaled std : {np.round(X_train_scaled.std(axis=0), 2)}")

print()
print("--- Feature Ranges (UI sliders ke liye) ---")
for col in iris.feature_names:
    print(f"{col}: min={df[col].min()} | max={df[col].max()}")


# ============================================================
# Section 3: KNN Algorithm + Elbow Method
# ============================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- Step 1: Elbow Method (Best K dhoondho) ---

error_rates = []

# K values 1 se 20 tak check karenge
k_range = range(1, 21)

for k in k_range:

    # Model banao
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train karo
    knn.fit(X_train_scaled, y_train)

    # Prediction lo
    y_pred_k = knn.predict(X_test_scaled)

    # Error calculate karo
    error = np.mean(y_pred_k != y_test)

    # Save karo
    error_rates.append(error)

# --- Error rates print karo ---
print("="*50)
print("ELBOW METHOD RESULTS")
print("="*50)

for k, err in zip(k_range, error_rates):
    print(f"K={k} --> Error Rate={err:.4f}")

# --- Elbow Graph Plot karo ---

plt.figure(figsize=(10, 5))

plt.plot(
    k_range,
    error_rates,
    marker='o',
    linestyle='--'
)

plt.title("Elbow Method For Optimal K")
plt.xlabel("K Value")
plt.ylabel("Error Rate")

plt.xticks(k_range)

plt.grid(True)

plt.show()



# ============================================================
# Final KNN Model
# ============================================================

while True:

    BEST_K = int(input("Enter stable K value (1-20): "))

    if 1 <= BEST_K <= 20:
        break

    print("Invalid K. Try again.")

# --- Final model banao ---
model = KNeighborsClassifier(n_neighbors=BEST_K)

# --- Training ---
model.fit(X_train_scaled, y_train)

# --- Prediction ---
y_pred = model.predict(X_test_scaled)

# --- Accuracy ---
accuracy = accuracy_score(y_test, y_pred)

print()
print("="*50)
print("FINAL MODEL PERFORMANCE")
print("="*50)

print(f"Best K Value : {BEST_K}")
print(f"Accuracy     : {accuracy:.4f}")
# ============================================================
# Section 4: Model Diagnostics
# ============================================================

# --- Precision ---
precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

# --- Recall ---
recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

# --- F1 Score ---
f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)

# --- Classification Report ---
report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
)

print()
print("="*50)
print("AI DIAGNOSTIC REPORT")
print("="*50)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print()
print("="*50)
print("CONFUSION MATRIX")
print("="*50)

print(cm)

print()
print("="*50)
print("CLASSIFICATION REPORT")
print("="*50)

print(report)


# ============================================================
# Flower Label Mapping
# ============================================================

flower_names = {
    0: "🌺 Iris Setosa",
    1: "🌸 Iris Versicolor",
    2: "🌼 Iris Virginica"
}

# ============================================================
# User Input System
# ============================================================

def get_user_input():

    print()
    print("="*50)
    print("ENTER FLOWER DIMENSIONS")
    print("="*50)

    sepal_length = float(input("Sepal Length (4.3 - 7.9): "))
    sepal_width  = float(input("Sepal Width  (2.0 - 4.4): "))
    petal_length = float(input("Petal Length (1.0 - 6.9): "))
    petal_width  = float(input("Petal Width  (0.1 - 2.5): "))

    user_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    return user_data

# ============================================================
# Anomaly Detection System
# ============================================================

def detect_anomaly(data):

    limits = {
        "sepal_length": (4.0, 8.0),
        "sepal_width":  (2.0, 5.0),
        "petal_length": (1.0, 7.0),
        "petal_width":  (0.1, 3.0)
    }

    values = data[0]

    for i, (feature, (low, high)) in enumerate(limits.items()):

        if values[i] < low or values[i] > high:

            print()
            print("⚠️  ANOMALY DETECTED")
            print("Alien Flora suspected.")
            print("Prediction aborted.")
            return True

    return False
# ============================================================
# Dev Matrix Terminal (UPGRADED)
# ============================================================

def dev_logs(user_scaled_data, X_train_data, y_train_data, k_val):

    print()
    print("="*50)
    print("DEV MATRIX TERMINAL [Under the Hood]")
    print("="*50)

    print("> Initializing KNN prediction engine...")
    print(f"> Active K value: {k_val}")
    print("> Calculating Euclidean Distances: sqrt(Σ(Xi - Yi)²)")
    
    # Asli math calculate kar rahe hain (Euclidean Distance)
    distances = np.linalg.norm(X_train_data - user_scaled_data, axis=1)
    nearest_indices = np.argsort(distances)[:k_val]
    
    print(f"> Top {k_val} Nearest Neighbors Found:")
    for i, idx in enumerate(nearest_indices):
        dist = distances[idx]
        label = flower_names[y_train_data[idx]]
        print(f"   [{i+1}] Distance: {dist:.4f} --> Class: {label}")
        
    print("> Running majority vote system...")
# ============================================================
# Prediction Engine
# ============================================================

# --- Input lo ---
user_flower = get_user_input()

# --- Anomaly detect karo ---
is_alien = detect_anomaly(user_flower)

# --- Agar anoma
# ly nahi hai toh continue ---
if not is_alien:

    # Scaling
    user_scaled = scaler.transform(user_flower)

    #dev logs with data
    dev_logs(user_scaled, X_train_scaled, y_train, BEST_K)

    # Prediction
    prediction = model.predict(user_scaled)[0]

    # Flower name
    flower = flower_names[prediction]

    print()
    print("="*50)
    print("FINAL PREDICTION")
    print("="*50)

    print(f"Predicted Flower: {flower}") 
    # XAI Heatmap
else:
    print("System halted due to anomaly.")    





# ============================================================
# Explainable AI (XAI) System
# ============================================================

def show_feature_importance(user_input):

    # Absolute scaled values
    importance = np.abs(user_input[0])

    # Normalize to percentage
    importance_percent = (
        importance / importance.sum()
    ) * 100

    print()
    print("="*50)
    print("FEATURE IMPORTANCE (XAI)")
    print("="*50)

    for name, score in zip(feature_names, importance_percent):

        print(f"{name:<20} --> {score:.2f}%")

    print("\n" + "="*50)
    print("THEORETICAL ANALYSIS (Why this prediction?)")
    print("="*50)
    
    highest_feature_idx = np.argmax(importance_percent)
    highest_feature_name = feature_names[highest_feature_idx]
    
    # Simple rule-based reasoning (Decision Tree logic on top of KNN)
    if prediction == 0: # Setosa
        print(f"> The AI has classified this as 'Setosa' primarily because of its {highest_feature_name}.")
        print("> Scientifically, Setosa flowers have extremely small Petal Lengths (typically under 2.0 cm).")
        print(f"> Your input perfectly aligns with this unique biological signature.")
    elif prediction == 1: # Versicolor
        print(f"> Classified as 'Versicolor'. This species acts as a bridge between Setosa and Virginica.")
        print(f"> The model heavily weighted {highest_feature_name} to distinguish it from the extremes.")
    else: # Virginica
        print(f"> Classified as 'Virginica'. The determining factor was {highest_feature_name}.")
        print("> Virginica species possess exceptionally large petals and sepals, breaking the threshold of common Iris flowers.")

    # --- Graph ---
    plt.figure(figsize=(10, 5))

    plt.bar(
        feature_names,
        importance_percent
    )

    plt.title("Feature Importance Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Importance (%)")

    plt.xticks(rotation=15)

    plt.grid(True)

    plt.show()  

if not is_alien:
    show_feature_importance(user_scaled)
    
else:
    print("XAI Heatmap skipped due to anomaly.")       
