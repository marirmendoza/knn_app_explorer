# 🔍 kNN Explorer App — Visualizing the Intuition Behind k-Nearest Neighbors

This interactive application, built with **Streamlit** and **Python**, allows you to explore the main concepts of the **k-Nearest Neighbors (kNN)** algorithm in a visual and intuitive way.

kNN is one of the simplest Machine Learning methods, yet it can model non-linear decision boundaries (as you will see for yourself).

This example focuses on kNN applied to **classification tasks** and was developed by **Prof. Mariana Recamonde Mendoza** as supporting material for the **Machine Learning** course taught at the **Institute of Informatics — Federal University of Rio Grande do Sul (UFRGS)**.

🔗 [https://inf-knn-app-explorer.streamlit.app](https://inf-knn-explorer-app.streamlit.app)

---

## App Goal

kNN has a very simple inductive bias:

> **Points with similar attributes values tend to be close in the feature space.**

This explorer helps you visualize:

- How the choice of **k** changes the decision boundary  
- The effect of different **distance metrics** (for now, we are comparing Euclidian and Manhatan distances) 
- How **feature scaling** influences the results  
- The importance of **feature normalization** for kNN
- How the algorithm behaves with **new, unseen data points**  
- The geometry of the data (**moons, blobs**), how it influences decisions, and how non-linear decision boundaries are formed  

This tool is intended for to serve as a support for lectures, self-study, and live demonstrations.

---

## App Overview

The application has three main sections:

1. **Settings (sidebar)**  
2. **Decision boundary visualization**  
3. **Testing with new points (generalization)**  

---

## kNN Model Settings

In the sidebar, you can adjust:

- **k (number of neighbors)**  
- **Distance metric** (Euclidean or Manhattan)  
- **Dataset** (Moons or Blobs)  
- **Normalization** (Min-Max scaling)  
- **Random seed for test points**  
- **Generate new test points**

These options allow you to create small experiments and immediately observe the effect of modeling decisions.

---

## Exploratory Scenarios

The app includes two main scenarios:

### 1️⃣ Local Boundary

This scenario allows you to observe:

- How a small k produces irregular boundaries (tendency toward high variance)  
- How a large k smooths the boundary (tendency toward high bias)  

### 2️⃣ Impact of Scale

This scenario demonstrates that:

- Features with large values dominate the distance computation  
- Normalizing features (for example, using Min-Max scaling) is essential in kNN  
- Without normalization, the axis with the largest range “drives” the decision  

---

## Testing with Unseen Points

The app automatically generates **10 new points** (using their own random seed), which simulate unseen, new data.

You can:

- Select a point (1 to 10)  
- Compare the **true class** and the **predicted class** for this point
- See the selected point highlighted in the plot  
- View all numbered points in the feature space  

This makes it clearer how the model behaves when making predictions on new data.



---

## Credits

**Author:** Prof. Mariana Recamonde Mendoza  [Personal website](https://www.inf.ufrgs.br/~mrmendoza/)
[Institute of Informatics](https://www.inf.ufrgs.br/site/) - Federal University of Rio Grande do Sul (UFRGS)


---
## Notes
*The code was developed with the support of Generative AI (Gemini 3.1 and ChatGPT 5.2).*
