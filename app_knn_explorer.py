import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# INSTITUTIONAL HEADER
# ============================================================

st.markdown("""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #004a99;">
    <strong>Machine Learning – Prof. Mariana Recamonde Mendoza</strong><br>
    Institute of Informatics, Federal University of Rio Grande do Sul (UFRGS).<br>
    <em>Interactive material developed with the support of generative AI (ChatGPT + Gemini).</em>
</div>
""", unsafe_allow_html=True)

st.title("🔍 Interactive kNN Explorer — Intuition, Neighborhood, and Scale")

st.markdown("""
The **k-Nearest Neighbors (kNN)** algorithm assumes that similar points should be close in feature space.

This explorer allows you to interactively visualize:

- how the value of **k** affects the decision boundary
- the dramatic effect of **feature scaling**
- the impact of **normalization**
- a simple **generalization** test with new unseen points
""")


# ============================================================
# SIDEBAR – SETTINGS
# ============================================================

st.sidebar.header("🛠️ kNN Settings")
k = st.sidebar.slider("Value of k (Neighbors)", 1, 31, 3, step=2)
metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan"])

st.sidebar.markdown("---")

scenario = st.sidebar.radio("Exploration Scenario:", [
    "Local Boundary (k=1 vs k=25)",
    "Impact of Scale",
])

dataset_type = st.sidebar.selectbox("Dataset", ["Moons", "Blobs"])

normalize = st.sidebar.checkbox("Enable Normalization", value=False)


# ============================================================
# GENERATE TEST POINTS (ON START OR DATASET CHANGE)
# ============================================================

def generate_test_points(dataset, seed):
    if dataset == "Moons":
        return make_moons(n_samples=10, noise=0.25, random_state=seed)
    else:
        return make_blobs(
            n_samples=10,
            centers=2,
            cluster_std=1.2,
            random_state=seed
        )

if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = dataset_type

if "test_seed" not in st.session_state:
    st.session_state.test_seed = 42

if "test_points" not in st.session_state:
    st.session_state.test_points = generate_test_points(dataset_type, st.session_state.test_seed)

# Regenerate when dataset changes
if dataset_type != st.session_state.current_dataset:
    st.session_state.current_dataset = dataset_type
    st.session_state.test_points = generate_test_points(dataset_type, st.session_state.test_seed)


# ============================================================
# BUTTON TO GENERATE NEW TEST POINTS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Generate New Test Points")

seed_input = st.sidebar.text_input("Random seed:", value=str(st.session_state.test_seed))

if st.sidebar.button("Generate New Points"):
    try:
        new_seed = int(seed_input)
        st.session_state.test_seed = new_seed
        st.session_state.test_points = generate_test_points(dataset_type, new_seed)
        st.sidebar.success(f"New points generated with seed = {new_seed}")
    except:
        st.sidebar.error("Invalid seed. Please enter an integer.")


# ============================================================
# 1. PERSISTENT BASE DATASET
# ============================================================

def generate_base_data(dataset):
    if dataset == "Moons":
        return make_moons(n_samples=300, noise=0.20, random_state=42)
    else:
        return make_blobs(n_samples=300, centers=2, cluster_std=1.2, random_state=42)

X_base, y_base = generate_base_data(dataset_type)
X = X_base.copy()
y = y_base.copy()


# ============================================================
# 2. APPLY SCENARIO
# ============================================================

if scenario == "Local Boundary (k=1 vs k=25)":
    info = (
        "With k=1, small 'islands' appear — local overfitting. "
        "With k=25, the boundary becomes much smoother."
    )

elif scenario == "Impact of Scale":
    X[:, 1] *= 50
    info = (
        "The Y-axis was multiplied by 50 — without normalization, "
        "vertical distance completely dominates."
    )


# ============================================================
# 3. NORMALIZATION
# ============================================================

if normalize:
    scaler = MinMaxScaler()
    X_model = scaler.fit_transform(X)
else:
    X_model = X


# ============================================================
# 4. MODEL TRAINING
# ============================================================

clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
clf.fit(X_model, y)


# ============================================================
# 5. GENERATE DECISION BOUNDARY
# ============================================================

h = 0.1
x_min, x_max = X_model[:, 0].min() - 0.5, X_model[:, 0].max() + 0.5
y_min, y_max = X_model[:, 1].min() - 0.5, X_model[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h),
)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)


# ============================================================
# 6. MAIN VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
ax.scatter(X_model[:, 0], X_model[:, 1], c=y, edgecolors='k', cmap='RdBu', alpha=0.8)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title(f"Decision Boundary (k={k}, Metric={metric})")
st.pyplot(fig)

st.info(f"**Insight:** {info}")


# ============================================================
# 7. TEST WITH UNSEEN POINTS
# ============================================================

st.markdown("---")
st.subheader("📊 Test with Unseen Points")

X_test_raw, y_test = st.session_state.test_points
X_test = scaler.transform(X_test_raw) if normalize else X_test_raw

point_idx = st.selectbox(
    "Select test point:",
    range(10),
    format_func=lambda x: f"Point {x+1}"
)

test_point = X_test[point_idx].reshape(1, -1)
pred = clf.predict(test_point)[0]
real = y_test[point_idx]

status = "✅ CORRECT" if pred == real else "❌ ERROR"
st.metric("Prediction Result", status)
st.write(f"**Predicted Class:** {pred}")
st.write(f"**True Class:** {real}")


# ============================================================
# VISUALIZATION OF TEST POINTS WITH IDENTIFICATION 1–10
# ============================================================

fig_test, ax_test = plt.subplots(figsize=(8, 4))
ax_test.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
ax_test.scatter(X_model[:, 0], X_model[:, 1], c=y, cmap='RdBu', alpha=0.3)

for i in range(10):
    px, py = X_test[i]

    if i == point_idx:
        ax_test.scatter(px, py, s=200, marker='X', c='yellow', edgecolors='black')
    else:
        ax_test.scatter(px, py, s=80, marker='o', c='black', edgecolors='white')

    # Correct numbering: 1–10
    ax_test.text(
        px + 0.05, py + 0.05,
        str(i + 1),
        fontsize=10,
        color='yellow' if i == point_idx else 'white',
        bbox=dict(facecolor='black', alpha=0.4, edgecolor='none')
    )

ax_test.set_xlabel("Feature 1")
ax_test.set_ylabel("Feature 2")
ax_test.set_title("Test Points (identification 1–10)")
st.pyplot(fig_test)


# ============================================================
# ACCURACY ON THE 10 POINTS
# ============================================================

acc = np.mean(clf.predict(X_test) == y_test)
st.write(f"**Correct classification rate on the 10 Points:** `{acc:.0%}`")
