import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import MinMaxScaler

# --- CABEÇALHO INSTITUCIONAL ---
st.markdown("""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #004a99;">
    <strong>Aprendizado de Máquina - Profa. Mariana Recamonde Mendoza.</strong><br>
    Instituto de Informática, Universidade Federal do Rio Grande do Sul (UFRGS).<br>
    <em>Este recurso foi criado com o apoio de IA generativa — Gemini 3.1 Pro.</em>
</div>
""", unsafe_allow_html=True)

st.title("🧩 Explorador kNN: A Intuição da Vizinhança")

st.markdown("""
O k-Nearest Neighbors (kNN) baseia-se em uma suposição simples: **pontos semelhantes tendem a estar próximos no espaço.** 
Chamamos isso de **Viés Indutivo** de suavidade local.
""")

# --- CONTROLES NA BARRA LATERAL ---
st.sidebar.header("🛠️ Configurações do kNN")
k = st.sidebar.slider("Valor de k (Vizinhos)", 1, 31, 3, step=2)
metric = st.sidebar.selectbox("Métrica de Distância", ["euclidean", "manhattan"])

st.sidebar.markdown("---")
scenario = st.sidebar.radio("Cenário de Exploração:", 
                            ["Fronteira Local (k=1 vs k=25)", 
                             "Impacto da Escala", 
                             "Diferentes Métricas"])

# --- GERAÇÃO DE DADOS ---
if scenario == "Fronteira Local (k=1 vs k=25)":
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    info = "Note como K=1 cria 'ilhas' ao redor de cada ponto (overfitting local), enquanto K=25 torna a fronteira mais suave."
elif scenario == "Impacto da Escala":
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=42)
    X[:, 1] = X[:, 1] * 50 # Distorce o eixo Y
    info = "O eixo Y está em uma escala 50x maior que o X. Note como a distância 'horizontal' perde importância sem normalização!"
else: # Diferentes Métricas
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=2.0, random_state=10)
    info = "A distância Euclidiana (linha reta) gera fronteiras circulares. A Manhattan (blocos) gera fronteiras mais retangulares/diamantes."

# --- OPÇÃO DE NORMALIZAÇÃO ---
normalize = st.sidebar.checkbox("Ativar Normalização", value=False)
if normalize:
    scaler = MinMaxScaler()
    X_model = scaler.fit_transform(X)
else:
    X_model = X

# --- MODELAGEM ---
clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
clf.fit(X_model, y)

# --- VISUALIZAÇÃO ---
h = .1
x_min, x_max = X_model[:, 0].min() - 0.5, X_model[:, 0].max() + 0.5
y_min, y_max = X_model[:, 1].min() - 0.5, X_model[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
ax.scatter(X_model[:, 0], X_model[:, 1], c=y, edgecolors='k', cmap='RdBu', alpha=0.8)
ax.set_title(f"Fronteira de Decisão (k={k}, Métrica={metric})")
st.pyplot(fig)

st.info(f"**Insight:** {info}")

# --- INTUIÇÃO DE AVALIAÇÃO ---
st.markdown("---")
st.subheader("📊 Como saber se o modelo está bom?")
st.write("""
Nesta introdução, não usaremos fórmulas complexas. Imagine que guardamos 10 pontos que o modelo nunca viu. 
A **Taxa de Acerto** seria simplesmente: quantos desses 10 o modelo classificou corretamente ao olhar para seus vizinhos?
""")
