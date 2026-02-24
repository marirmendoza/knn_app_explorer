import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# CABEÇALHO INSTITUCIONAL
# ============================================================

st.markdown("""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #004a99;">
    <strong>Aprendizado de Máquina – Profa. Mariana Recamonde Mendoza</strong><br>
    Instituto de Informática, Universidade Federal do Rio Grande do Sul (UFRGS).<br>
    <em>Material interativo desenvolvido com apoio de IA generativa (ChatGPT + Gemini).</em>
</div>
""", unsafe_allow_html=True)

st.title("🔍 Explorador Interativo do kNN — Intuição, Vizinhança e Escala")

st.markdown("""
O algoritmo **k-Nearest Neighbors (kNN)** assume que:

> **Pontos semelhantes tendem a estar próximos no espaço.**

Este explorador mostra, de forma visual, o efeito:
- do valor de **k**
- da **escala dos atributos**
- das diferentes **métricas de distância**
- da **normalização**
- e da **generalização com novos pontos de teste**
""")

# ============================================================
# SIDEBAR – CONFIGURAÇÕES
# ============================================================

st.sidebar.header("🛠️ Configurações do kNN")
k = st.sidebar.slider("Valor de k (Vizinhos)", 1, 31, 3, step=2)
metric = st.sidebar.selectbox("Métrica de Distância", ["euclidean", "manhattan"])

st.sidebar.markdown("---")

scenario = st.sidebar.radio("Cenário de Exploração:", [
    "Fronteira Local (k=1 vs k=25)",
    "Impacto da Escala",
    "Diferentes Métricas"
])

dataset_type = st.sidebar.selectbox("Base de Dados", ["Moons", "Blobs"])

normalize = st.sidebar.checkbox("Ativar Normalização", value=False)

# ============================================================
# GERAR TESTE QUANDO DATASET MUDA
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

# Inicializar session_state
if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = dataset_type
if "test_seed" not in st.session_state:
    st.session_state.test_seed = 42
if "test_points" not in st.session_state:
    st.session_state.test_points = generate_test_points(dataset_type, st.session_state.test_seed)

# Regenerar quando dataset muda
if dataset_type != st.session_state.current_dataset:
    st.session_state.current_dataset = dataset_type
    st.session_state.test_points = generate_test_points(dataset_type, st.session_state.test_seed)


# ============================================================
# BOTÃO PARA GERAR NOVOS PONTOS DE TESTE
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Gerar novos pontos de teste")

seed_input = st.sidebar.text_input("Seed aleatória:", value=str(st.session_state.test_seed))

if st.sidebar.button("Gerar novos pontos"):
    try:
        new_seed = int(seed_input)
        st.session_state.test_seed = new_seed
        st.session_state.test_points = generate_test_points(dataset_type, new_seed)
        st.sidebar.success(f"Novos pontos gerados com seed = {new_seed}")
    except:
        st.sidebar.error("Seed inválida. Use um número inteiro.")


# ============================================================
# 1. DATASET BASE PERSISTENTE
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
# 2. APLICAÇÃO DO CENÁRIO
# ============================================================

if scenario == "Fronteira Local (k=1 vs k=25)":
    info = (
        "Com k=1 surgem pequenas 'ilhas' — overfitting local. "
        "Com k=25 a fronteira fica muito mais suave."
    )

elif scenario == "Impacto da Escala":
    X[:, 1] *= 50
    info = (
        "O eixo Y foi multiplicado por 50 — sem normalização "
        "a distância vertical domina completamente."
    )

elif scenario == "Diferentes Métricas":
    if dataset_type == "Moons":
        X[:, 0] *= 2
        X[:, 1] *= 0.5
    else:
        X, y = make_blobs(n_samples=300, centers=2, cluster_std=2.0, random_state=10)
    info = (
        "Euclidiana gera fronteiras circulares; Manhattan gera regiões losangulares."
    )


# ============================================================
# 3. NORMALIZAÇÃO
# ============================================================

if normalize:
    scaler = MinMaxScaler()
    X_model = scaler.fit_transform(X)
else:
    X_model = X


# ============================================================
# 4. TREINO DO MODELO
# ============================================================

clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
clf.fit(X_model, y)


# ============================================================
# 5. GERAÇÃO DA FRONTEIRA
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
# 6. VISUALIZAÇÃO PRINCIPAL
# ============================================================

st.markdown("---")
st.subheader("🔍 Visualização da Fronteira de Decisão")


fig, ax = plt.subplots(figsize=(10, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
ax.scatter(X_model[:, 0], X_model[:, 1], c=y, edgecolors='k', cmap='RdBu', alpha=0.8)

ax.set_title(f"Fronteira de Decisão (k={k}, Métrica={metric})")
st.pyplot(fig)

st.info(f"**Insight:** {info}")


# ============================================================
# 7. TESTANDO PONTOS DESCONHECIDOS
# ============================================================

st.markdown("---")
st.subheader("📊 Teste com Pontos Desconhecidos")

X_test_raw, y_test = st.session_state.test_points
# Normalização coerente
X_test = scaler.transform(X_test_raw) if normalize else X_test_raw

point_idx = st.selectbox("Selecione o ponto de teste:", range(10), format_func=lambda x: f"Ponto {x+1}")

test_point = X_test[point_idx].reshape(1, -1)
pred = clf.predict(test_point)[0]
real = y_test[point_idx]

status = "✅ ACERTO" if pred == real else "❌ ERRO"
st.metric("Resultado da Predição", status)
st.write(f"**Classe Predita:** {pred}")
st.write(f"**Classe Real:** {real}")


# ============================================================
# VISUALIZAÇÃO INTERATIVA COM IDENTIFICAÇÃO DOS PONTOS
# ============================================================

fig_test, ax_test = plt.subplots(figsize=(8, 4))
ax_test.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
ax_test.scatter(X_model[:, 0], X_model[:, 1], c=y, cmap='RdBu', alpha=0.3)

# PLOTS COM NÚMEROS EXPLICATIVOS
for i in range(10):
    px, py = X_test[i]
    if i == point_idx:
        ax_test.scatter(px, py, s=200, marker='X', c='yellow', edgecolors='black')
    else:
        ax_test.scatter(px, py, s=80, marker='o', c='black', edgecolors='white')
    
    # Adiciona número do ponto ao lado
    ax_test.text(px + 0.05, py + 0.05, str(i+1), fontsize=10, color='yellow' if i==point_idx else 'white',
                 bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))

ax_test.set_title("Pontos de Teste (numeração visível)")
st.pyplot(fig_test)


# ============================================================
# ACURÁCIA DOS 10 PONTOS
# ============================================================

acc = np.mean(clf.predict(X_test) == y_test)
st.write(f"**Taxa de Acerto nos 10 Pontos de Teste:** `{acc:.0%}`")
