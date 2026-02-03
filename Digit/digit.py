from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load data
digits = load_digits()

X = digits.data
y = digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Best Pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma=0.01,
        probability=True
    ))
])

# Train
model.fit(X_train, y_train)

# Check accuracy
acc = model.score(X_test, y_test)
print("Accuracy:", acc)

# Save
joblib.dump(model, "digit_best.pkl")