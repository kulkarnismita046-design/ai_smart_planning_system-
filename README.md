# ai_smart_planning_system-
Ai smart planning system AI Smart Planning System  (BAI12304 - Advanced Python Programming Lab)
Uses: OOP, NumPy, Pandas, Matplotlib/Seaborn, Linear Regression,
      Logistic Regression, SVM, k-NN.

Pipeline:
  1. Build a task dataset (Pandas + NumPy)
  2. EDA + preprocessing
  3. Linear Regression  -> predict task duration (minutes)
  4. Logistic Regression / SVM / k-NN -> predict priority (Low/Med/High), compare models
  5. Planner class -> schedule the day using the AI predictions
  6. Visualize: model comparison, confusion matrix, Gantt-style schedule

Run: python ai_smart_planner_lab.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score,
                             classification_report, confusion_matrix)

rng = np.random.default_rng(42)
PRIORITY = {0: "Low", 1: "Medium", 2: "High"}
CATEGORIES = ["study", "work", "health", "personal"]


# ---------------------------------------------------------------- 1. DATA
def make_dataset(n=300):
    df = pd.DataFrame({
        "hours_to_deadline": rng.uniform(1, 72, n).round(1),
        "importance": rng.integers(1, 6, n),
        "complexity": rng.integers(1, 11, n),
        "category": rng.choice(CATEGORIES, n),
    })
    # true duration depends on complexity (+ noise) -> target for regression
    df["duration_min"] = (12 * df.complexity + 8 * (df.category == "study")
                          + rng.normal(0, 8, n)).clip(5).round()
    # priority score: important + close deadline -> high priority (+ noise)
    score = 2.0 * df.importance + 30 / df.hours_to_deadline + rng.normal(0, 1.2, n)
    df["priority"] = pd.cut(score, [-np.inf, 5.5, 8.5, np.inf], labels=[0, 1, 2]).astype(int)
    return df


df = make_dataset()
print("=== Dataset preview ===")
print(df.head(), "\n")
print(df.describe().round(2), "\n")
print("Missing values:", int(df.isna().sum().sum()))
print("Priority counts:\n", df.priority.map(PRIORITY).value_counts(), "\n")

# ---------------------------------------------------------- 2. PREPROCESSING
X = pd.get_dummies(df[["hours_to_deadline", "importance", "complexity", "category"]],
                   columns=["category"], dtype=int)
FEATURES = list(X.columns)
X_tr, X_te, yd_tr, yd_te, yp_tr, yp_te = train_test_split(
    X, df.duration_min, df.priority, test_size=0.25, random_state=42, stratify=df.priority)
scaler = StandardScaler().fit(X_tr)
Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

# ------------------------------------------- 3. LINEAR REGRESSION (duration)
dur_model = LinearRegression().fit(Xs_tr, yd_tr)
pred = dur_model.predict(Xs_te)
print("=== Linear Regression: task duration ===")
print(f"MAE = {mean_absolute_error(yd_te, pred):.2f} min | R2 = {r2_score(yd_te, pred):.3f}\n")

# ------------------------- 4. CLASSIFIERS (priority): LogReg vs SVM vs k-NN
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel="rbf", C=2.0),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
}
acc = {}
fitted = {}
for name, m in models.items():
    m.fit(Xs_tr, yp_tr)
    p = m.predict(Xs_te)
    acc[name] = accuracy_score(yp_te, p)
    fitted[name] = (m, p)
    print(f"=== {name} === accuracy = {acc[name]:.3f}")
    print(classification_report(yp_te, p, target_names=list(PRIORITY.values()), zero_division=0))
best = max(acc, key=acc.get)
prio_model, best_pred = fitted[best]
print(f"Best model for priority: {best}\n")

# ---------------------------------------------------------- 5. OOP PLANNER
class Task:
    def __init__(self, name, hours_to_deadline, importance, complexity, category):
        self.name, self.category = name, category
        self.hours_to_deadline, self.importance, self.complexity = hours_to_deadline, importance, complexity
        self.priority = None
        self.duration = None


class Planner:
    """Uses trained AI models to predict priority/duration, then schedules the day."""
    def __init__(self, dur_model, prio_model, scaler, features, start=9 * 60, end=17 * 60 + 30, brk=10):
        self.dm, self.pm, self.sc, self.features = dur_model, prio_model, scaler, features
        self.start, self.end, self.brk = start, end, brk

    def _vector(self, t):
        row = {f: 0 for f in self.features}
        row.update(hours_to_deadline=t.hours_to_deadline, importance=t.importance,
                   complexity=t.complexity)
        row[f"category_{t.category}"] = 1
        return pd.DataFrame([row])[self.features]

    def predict(self, tasks):
        for t in tasks:
            v = self.sc.transform(self._vector(t))
            t.priority = int(self.pm.predict(v)[0])
            t.duration = int(max(5, round(self.dm.predict(v)[0] / 5) * 5))

    def schedule(self, tasks):
        self.predict(tasks)
        order = sorted(tasks, key=lambda t: (-t.priority, t.hours_to_deadline))
        clock, plan, dropped = self.start, [], []
        for t in order:
            if clock + t.duration > self.end:
                dropped.append(t.name)
                continue
            plan.append((t, clock, clock + t.duration))
            clock += t.duration + self.brk
        return plan, dropped


def hhmm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


planner = Planner(dur_model, prio_model, scaler, FEATURES)
todays_tasks = [
    Task("Finish lab report", 6, 5, 8, "study"),
    Task("Reply to emails", 20, 3, 2, "work"),
    Task("Gym workout", 30, 2, 4, "health"),
    Task("Prepare presentation", 4, 5, 9, "work"),
    Task("Pay bills", 3, 4, 1, "personal"),
    Task("Revise for exam", 12, 4, 10, "study"),
    Task("Organize photos", 60, 1, 5, "personal"),
]
plan, dropped = planner.schedule(todays_tasks)
print("=== AI Smart Plan ===")
for t, a, b in plan:
    print(f"{hhmm(a)}-{hhmm(b)}  [{PRIORITY[t.priority]:6}] {t.name} ({t.duration} min)")
if dropped:
    print("Not enough time today:", ", ".join(dropped))

# ---------------------------------------------------------- 6. VISUALIZATION
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))

sns.barplot(x=list(acc), y=list(acc.values()), ax=ax[0], hue=list(acc), legend=False)
ax[0].set(ylim=(0, 1), title="Priority model accuracy", ylabel="Accuracy")
ax[0].tick_params(axis="x", rotation=15)

sns.heatmap(confusion_matrix(yp_te, best_pred), annot=True, fmt="d", cmap="Blues", ax=ax[1],
            xticklabels=PRIORITY.values(), yticklabels=PRIORITY.values())
ax[1].set(title=f"Confusion matrix - {best}", xlabel="Predicted", ylabel="Actual")

ax[2].scatter(yd_te, pred, alpha=.6)
lim = [yd_te.min(), yd_te.max()]
ax[2].plot(lim, lim, "r--")
ax[2].set(title="Linear Regression: actual vs predicted duration", xlabel="Actual (min)", ylabel="Predicted (min)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/model_results.png", dpi=130)
plt.close()

colors = {0: "#2f9e44", 1: "#f59f00", 2: "#e8590c"}
fig, ax = plt.subplots(figsize=(10, 4.5))
for i, (t, a, b) in enumerate(plan):
    ax.barh(i, b - a, left=a, color=colors[t.priority])
    ax.text(a + 2, i, f"{t.name} ({t.duration}m)", va="center", color="white", fontsize=9, fontweight="bold")
ticks = np.arange(planner.start, planner.end + 1, 60)
ax.set_xticks(ticks, [hhmm(x) for x in ticks])
ax.set_yticks([])
ax.invert_yaxis()
ax.set_title("AI-generated daily schedule (green=low, orange=medium, red=high priority)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/daily_schedule.png", dpi=130)
print("\nSaved: model_results.png, daily_schedule.png")

