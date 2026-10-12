# Bloc C — Comparaison Critique : Interprétation Humaine vs IA Générative
## Évaluation Empirique et Grille d'Arbitrage Méthodologique

**Auteur de la comparaison :** Étudiante en Ingénierie Data Science & IA (TEK-UP)  
**Date d'évaluation :** 12 Octobre 2026  
**Référence Humaine :** [`results/llm/reference_interpretation.md`](file:///c:/Users/kenza/Desktop/Financial_Time_Series_Restart/results/llm/reference_interpretation.md)  
**Modèle Évalué :** `openai/gpt-oss-120b` (via Groq API, exécuté le 12/10/2026)  
**Corpus de réponses analysé :** [`results/llm/experiments/`](file:///c:/Users/kenza/Desktop/Financial_Time_Series_Restart/results/llm/experiments/) (`response_1`, `response_2`, `response_3`)  

---

### 1. Protocole et Barème Reproductible d'Évaluation

Pour évaluer de manière objective et transparente la performance de l'IA générative face au benchmark étudiant, une grille d'évaluation à 6 dimensions a été formalisée. Chaque dimension est notée sur une échelle de **1 à 5 points** selon une rubrique explicite :

- **1/5 — Insuffisant :** Erreurs conceptuelles majeures, données inventées ou contraintes violées.
- **2/5 — Passable :** Compréhension superficielle, raccourcis trompeurs, nuances méthodologiques omises.
- **3/5 — Satisfaisant :** Analyse globalement exacte mais manquant de précision ou de recul critique.
- **4/5 — Bon :** Analyse exacte, conforme aux données, intégrant l'incertitude avec quelques simplifications mineures.
- **5/5 — Excellent :** Rigueur mathématique irréprochable, fidélité totale aux données, nuances subtiles maîtrisées et prudence déontologique absolue.

---

### 2. Tableau Comparatif Multidimensionnel et Notations Justifiées

| Dimension Analysée | Note Humain (Étudiante) | Note LLM (`gpt-oss-120b`) | Analyse Comparative et Justification des Notes |
|---|:---:|:---:|---|
| **1. Exactitude Statistique & Définitions** | **5 / 5** | **4 / 5** | *Étudiante :* Distingue rigoureusement l'échec de rejet d'une racine unitaire (ADF) du rejet de la stationnarité (KPSS). <br>*LLM :* Énonce que les deux tests "s'accordent pour rejeter la stationnarité" (raccourci formel sur ADF). Recommande une VaR paramétrique sans mentionner initialement l'adaptation au quantile de Student. |
| **2. Fidélité Factuelle aux Données** | **5 / 5** | **4,5 / 5** | *Étudiante :* Intègre l'ensemble des 18 modèles des 3 actifs. <br>*LLM :* Reproduit parfaitement toutes les tables fournies sans aucune hallucination numérique. Seule réserve : conclut qu'aucun modèle n'a été estimé sur le S&P 500 car le prompt lui avait transmis un extrait tronqué (`comp[:12]`). |
| **3. Clarté & Vulgarisation Executive** | **4 / 5** | **5 / 5** | *Étudiante :* Rédaction académique très détaillée, parfois trop dense pour un décideur non-mathématicien. <br>*LLM :* Excellente synthèse exécutive structurée par tableaux synthétiques ("Ce que les modèles nous disent – résumé pour le CFO"). Très pédagogique sans trahir la réalité. |
| **4. Traitement de l'Incertitude & Limites** | **4,5 / 5** | **4,5 / 5** | *Étudiante :* Détaille les degrés de liberté, l'asymétrie de sélection d'ordre, et la perte de puissance liée à $N=25$. <br>*LLM :* Assimile et restitue avec une grande fidélité les 5 limites critiques (désaccord de Johansen, résidus VAR à $p=0.028$, DM post-Holm non significatif, brièveté du test). |
| **5. Prudence Déontologique & Recommandations** | **5 / 5** | **5 / 5** | *Ex aequo parfait :* Les deux refusent catégoriquement de promettre des gains de trading ou de l'alpha directionnel. Tous deux orientent l'usage vers la gestion du risque et le dimensionnement des limites de volatilité via GARCH. |
| **6. Résistance à l'Hallucination Causale** | **5 / 5** | **5 / 5** | *Ex aequo parfait :* Face au test de Granger S&P 500 $\to$ Gold ($p = 0.0013$), le modèle s'est conformé à la règle : il explicite qu'il ne s'agit en aucun cas d'une cause réelle ou mécanique, mais d'une simple corrélation prédictive temporelle. |
| **Score Total Consolidé** (somme des six dimensions) | **28,5 / 30 (95,0 %)** | **28,0 / 30 (93,3 %)** | **Conclusion :** L'étudiante l'emporte sur la rigueur formelle et la vision globale ; le LLM excelle dans la concision et la communication d'affaires. |

---

### 3. Analyse Détaillée des Écarts et Complémentarités

#### A. Là où l'expertise humaine surpasse l'IA :
1. **La conscience du contexte global du projet :**  
   L'étudiante sait que le S&P 500 a fait l'objet d'un modèle ARIMA(0,0,0) et d'un VAR(1) dans le Bloc A. Le LLM, restreint par la fenêtre d'information fournie dans le prompt 2, a conclu qu'aucun modèle n'existait sur cet actif. L'IA est fondamentalement limitée par la complétude du contexte injecté (*prompt-dependent myopia*).
2. **La rigueur sémantique des hypothèses de tests :**  
   En économétrie, la confusion entre *"ne pas rejeter $H_0$"* et *"prouver $H_1$"* est une faute méthodologique classique. L'étudiante évite ce piège sur le test ADF, alors que le modèle a utilisé une formulation simplifiée pour fluidifier son texte.
3. **L'adaptation fine de la formule de risque :**  
   Dans la note au comité des risques, le LLM propose $VaR = z_\alpha \cdot \hat{\sigma}_t$ en citant $z_{0.95}$, ce qui correspond à la loi Normale. L'étudiante rappelle que les queues de distribution sont épaisses ($\nu \approx 5$ à $8$) et qu'il faut utiliser le quantile de la loi de Student normalisée. Nuance exacte : au seuil de 95 %, le quantile gaussien 1,645 légèrement supérieur au quantile de Student standardisé (1,57–1,62) conduit à une VaR un peu conservatrice, tandis qu'aux seuils extrêmes (99 % et au-delà) l'usage du quantile gaussien sous-calibre le capital économique. La VaR paramétrique n'a pas été implémentée dans le projet ; il s'agit d'une proposition du LLM discutée critiquement.

#### B. Là où l'IA apporte une valeur ajoutée indiscutable :
1. **L'impact de la communication d'affaires :**  
   La réponse 2 du modèle démontre une capacité remarquable à vulgariser la théorie de l'efficience des marchés (*martingale*) pour un CFO en trois points clés : intégration immédiate de l'information, niveau de bruit dominant le signal, et risque de sur-apprentissage lié aux modèles complexes.
2. **La synthèse opérationnelle pour la gestion des risques :**  
   La procédure en trois étapes quotidiennes formulée dans la réponse 3 (estimation $\hat{\sigma}_t \to$ calcul VaR $\to$ vérification de l'exposition nette) constitue un livrable professionnel directement exploitable en salle des marchés.

---

### 4. Verdict pour le Mini-Projet

La confrontation empirique valide pleinement l'approche préconisée par l'enseignant :
- L'IA générative ne doit pas être utilisée comme un oracle autonome ou un rédacteur automatique.
- Utilisée sous le contrôle d'un Data Scientist qui fournit un cadre de données vérifiées et audite chaque réponse, elle devient un **accélérateur de synthèse et de vulgarisation de premier ordre**.
