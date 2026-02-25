# 🔍 App kNN Explorer — Visualizando a Intuição do k-Nearest Neighbors

Este aplicativo interativo em ***Streamlit*** e ***Python***  permite explorar, de maneira visual e intuitiva, os principais conceitos do algoritmo **k-Nearest Neighbors (kNN)** — um dos métodos mais simples em Aprendizado de Máquina, e capaz de modelar fronteiras de decisão não-lineares. 

Este exemplo abrange uma aplicação do kNN em tarefas de classificação, e foi desenvolvido como material de apoio para a disciplina de **Aprendizado de Máquina**, da **Profa. Mariana Recamonde Mendoza**, no **Instituto de Informática — Universidade Federal do Rio Grande do Sul (UFRGS)**

---

## Objetivo do Aplicativo

O kNN possui um viés indutivo muito simples:

> **Pontos com atributos semelhantes tendem a estar próximos no espaço.**

Esse explorador permite visualizar:

- Como a escolha de **k** altera a fronteira de decisão
- O efeito de diferentes **métricas de distância**  
- Como a diferença de **escala dos atributos** influencia o resultado  
- A importância da **normalização**  de atributos
- Como o algoritmo se comporta com **novos pontos nunca vistos**  
- A geometria dos dados **(moons, blobs)**, sua influência na decisão, e a formação de fronteiras de decisão não-lineares

É uma ferramenta ideal para aulas, estudos individuais e demonstrações ao vivo.

---

## Visão Geral do App

<img width="1124" height="710" alt="Captura de Tela 2026-02-24 às 19 39 52" src="https://github.com/user-attachments/assets/4cde5991-505d-497e-b3cf-db1d55cb51c9" />


O aplicativo possui três áreas principais:

1. **Configurações (barra lateral)**  
2. **Visualização da fronteira de decisão**  
3. **Teste com novos pontos (generalização)**  

---

## Configurações do Modelo kNN

Na barra lateral, você pode ajustar:

- **k (número de vizinhos)**  
- **Métrica de distância** (euclidiana ou manhattan)  
- **Base de dados** (Moons ou Blobs)  
- **Normalização** (Pelo método Min-Max)  
- **Seed aleatória dos pontos de teste**  
- **Gerar novos pontos de teste**

Essas opções permitem construir experimentos para visualizar, imediatamente, o efeito das decisões de modelagem.

---

## Cenários Exploratórios

O app possui dois cenários principais:

### 1️⃣ **Fronteira Local**
Permite observar:

- Como k pequeno gera fronteiras irregulares (tendência a uma alta variância)  
- Como k grande suaviza a fronteira (tendência a um alto viés)  

### 2️⃣ **Impacto da Escala**
Demonstra que:

- Atributos com valores muito grandes dominam a distância  
- Normalizar (por exemplo, usando Min-Max) é essencial em kNN  
- Sem normalização, o eixo com maior amplitude “manda” na decisão  

---

## Teste com Pontos Desconhecidos

O app gera automaticamente **10 novos pontos** (com sua própria seed).

Você pode:

- Selecionar um ponto (1 a 10)
- Ver **classe verdadeira** × **classe predita**
- Ver o ponto destacado no gráfico  
- Ver todos os pontos numerados no plano  

Isso torna mais claro como o modelo se comporta com novos dados.

---

## Créditos
**Autora:** Profa. Mariana Recamonde Mendoza, Instituto de Informática, Universidade Federal do Rio Grande do Sul (UFRGS)

Nota: O código foi desenvolvido com o apoio de Gemini e chatGPT.
