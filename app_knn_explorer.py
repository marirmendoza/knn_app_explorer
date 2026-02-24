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
O algoritmo **k-Nearest Neighbors (kNN)** se apoia em um princípio muito simples e poderoso:

> **Pontos semelhantes tendem a estar próximos no espaço.**

Esse é o chamado **Viés Indutivo de Suavidade Local**:  
se dois pontos têm atributos parecidos, espera-se que pertençam à mesma classe.  

Este explorador permite visualizar, de maneira totalmente interativa:

- O efeito do valor de **k**  
- Como a **escala dos atributos** muda completamente a fronteira  
- Como diferentes **métricas de distância** deformam as regiões de decisão  
- O impacto da **normalização**  
- E até um pequeno teste com pontos "desconhecidos"  
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
# 1. DATASET BASE PERSISTENTE
# ============================================================

def generate_base_data(dataset):
    if dataset == "Moons":
        return make_moons(n_samples=300, noise=0.20, random_state=42)
    else:
        return make_blobs(n_samples=300, centers=2, cluster_std=1.2, random_state=42)

# dataset é sempre o MESMO para todos os cenários
X_base, y_base = generate_base_data(dataset_type)

# trabalhamos sobre cópia, não sobre o original
X = X_base.copy()
y = y_base.copy()


# ============================================================
# 2. APLICAÇÃO DO CENÁRIO ESCOLHIDO
# ============================================================

if scenario == "Fronteira Local (k=1 vs k=25)":
    info = (
        "Com k=1 surgem pequenas 'ilhas' ao redor de cada amostra — "
        "**overfitting local**. Com k=25, a fronteira se torna muito mais suave."
    )

elif scenario == "Impacto da Escala":
    X[:, 1] *= 50
    info = (
        "O eixo Y foi multiplicado por 50 — sem normalização a distância "
        "vertical domina completamente a classificação."
    )

elif scenario == "Diferentes Métricas":
    if dataset_type == "Moons":
        X[:, 0] *= 2
        X[:, 1] *= 0.5
    else:
        X, y = make_blobs(
            n_samples=300,
            centers=2,
            cluster_std=2.0,
            random_state=10
        )
    info = (
        "A distância Euclidiana (L2) tende a gerar fronteiras circulares; "
        "a Manhattan (L1) cria fronteiras mais retangulares ou losangulares."
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
# 5. GERAÇÃO DO GRID PARA A FRONTEIRA
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

fig, ax = plt.subplots(figsize=(10, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
ax.scatter(X_model[:, 0], X_model[:, 1], c=y, edgecolors='k', cmap='RdBu', alpha=0.8)
ax.set_title(f"Fronteira de Decisão (k={k}, Métrica={metric})")

st.pyplot(fig)
st.info(f"**Insight:** {info}")


# ============================================================
# 7. TESTE COM PONTOS DESCONHECIDOS (Generalização)
# ============================================================

st.markdown("---")
st.subheader("📊 Teste com Pontos Desconhecidos")

def generate_test_points(dataset):
    if dataset == "Moons":
        return make_moons(n_samples=10, noise=0.25, random_state=7)
    else:
        return make_blobs(
            n_samples=10,
            centers=2,
            cluster_std=1.2,
            random_state=15
        )

if "test_points" not in st.session_state:
    st.session_state.test_points = generate_test_points(dataset_type)

X_test, y_test = st.session_state.test_points

point_idx = st.selectbox("Selecione o ponto de teste:", range(10), format_func=lambda x: f"Ponto {x+1}")

test_raw = X_test[point_idx].reshape(1, -1)
test_point = scaler.transform(test_raw) if normalize else test_raw

pred = clf.predict(test_point)[0]
real = y_test[point_idx]

status = "✅ ACERTO" if pred == real else "❌ ERRO"
st.metric("Resultado da Predição", status)
st.write(f"**Classe Predita:** {pred}")
st.write(f"**Classe Real:** {real}")


# Plot dos pontos de teste
fig_test, ax_test = plt.subplots(figsize=(8, 4))
ax_test.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
ax_test.scatter(X_model[:, 0], X_model[:, 1], c=y, cmap='RdBu', alpha=0.3)

for i in range(10):
    p = X_test[i].reshape(1, -1)
    if normalize:
        p = scaler.transform(p)

    ax_test.scatter(
        p[0, 0], p[0, 1],
        s=180 if i == point_idx else 60,
        marker='X' if i == point_idx else 'o',
        c='yellow' if i == point_idx else 'black',
        edgecolors='black'
    )

ax_test.set_title("Pontos de Teste (Desconhecidos)")
st.pyplot(fig_test)

# acurácia simples
test_model = scaler.transform(X_test) if normalize else X_test
acc = np.mean(clf.predict(test_model) == y_test)

st.write(f"**Taxa de Acerto nos 10 Pontos:** `{acc:.0%}`")
