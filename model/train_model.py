import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import sqlite3

print("✅ Training script started...")

# Load dataset
try:
    df = pd.read_csv("flights.csv")
    print(f"📁 Dataset loaded successfully — {len(df)} rows, {len(df.columns)} columns")
except FileNotFoundError:
    print("❌ flights.csv not found! Please place it in C:\\dynamic_airfare_app")
    exit()

# Encode categorical columns
cat_cols = ['airline', 'source_city', 'destination_city',
            'departure_time', 'arrival_time', 'stops', 'class']
le_dict = {}

for c in cat_cols:
    if c not in df.columns:
        print(f"⚠️ Column missing in dataset: {c}")
        exit()
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    le_dict[c] = le

print("🔠 Label encoding complete")

# Train model (fast mode)
X = df.drop(columns=['price', 'flight'])
y = df['price']

model = RandomForestRegressor(
    n_estimators=30,   # reduced for speed
    max_depth=10,
    n_jobs=-1,         # use all CPU cores
    random_state=42
)

print("🚀 Starting model training...")
model.fit(X, y)
print("🎯 Model training complete")

# Save model and encoders
joblib.dump(model, "model/price_model.pkl")
joblib.dump(le_dict, "model/label_encoders.pkl")
print("💾 Model and encoders saved to /model folder")

# Save dataset to SQLite (for reference)
conn = sqlite3.connect("flights.db")
df.to_sql("flights", conn, if_exists="replace", index=False)
conn.close()

print("✅ All done — Model trained and data stored in flights.db")
