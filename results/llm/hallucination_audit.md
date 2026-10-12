# Bloc C — Protocole et Matrice d'Audit des Hallucinations Statistiques
## Contrôle de Cohérence Factuelle et Intégrité des Sorties LLM Réelles

**Auteur de l'audit :** Étudiante en Ingénierie Data Science & IA (TEK-UP)  
**Date de l'audit :** 12 Octobre 2026  
**Fournisseur / Modèle audité :** Groq API — `openai/gpt-oss-120b`  
**Horodatage d'exécution :** 12 Octobre 2026, 13:47 UTC (Exp 1 & 2) et 13:51 UTC (Exp 3)  
**Manifeste d'exécution :** [`results/llm/experiments/experiments_manifest.json`](file:///c:/Users/kenza/Desktop/Financial_Time_Series_Restart/results/llm/experiments/experiments_manifest.json) (`status: COMPLETED`)  
**Référence de vérité terrain :** [`results/llm/reference_interpretation.md`](file:///c:/Users/kenza/Desktop/Financial_Time_Series_Restart/results/llm/reference_interpretation.md) et [`results/llm/financial_project_context.json`](file:///c:/Users/kenza/Desktop/Financial_Time_Series_Restart/results/llm/financial_project_context.json)  

---

### 1. Objectifs et Méthodologie d'Audit Empirique

L'intégration de l'IA générative dans un projet de modélisation financière exige une vérification rigoureuse assertion par assertion. Cet audit examine les réponses brutes générées par le modèle `openai/gpt-oss-120b` sur les trois expériences du Bloc C.

Chaque affirmation extraite des fichiers de réponses brutes est classée selon la taxonomie en 4 niveaux :
- **SUPPORTED** : Affirmation rigoureusement exacte et corroborée par les données du projet.
- **OVERSIMPLIFIED** : Affirmation globalement vraie mais comportant un raccourci de langage ou omettant une nuance statistique critique.
- **UNSUPPORTED** : Affirmation plausible mais non étayée par les données fournies dans le contexte.
- **CONTRADICTED** : Affirmation en contradiction directe avec les résultats empiriques réels du projet.

---

### 2. Synthèse Statistique Globale de l'Audit

Sur l'ensemble des trois réponses générées par le modèle :
- **Total des assertions auditées :** 16 assertions représentatives (E1.1–E1.5, E2.1–E2.6, E3.1–E3.5, détaillées dans les matrices des sections 3 à 5)
- **SUPPORTED :** 12 assertions (75,00 %)
- **OVERSIMPLIFIED :** 3 assertions (18,75 %)
- **UNSUPPORTED :** 0 assertion (0,00 %)
- **CONTRADICTED (artéfact de découpage du prompt) :** 1 assertion (6,25 %)

*Note de périmètre :* cet audit porte sur un échantillon d'assertions soigneusement sélectionnées parmi les trois réponses (environ 18 500 caractères générés) et non sur une vérification exhaustive ligne par ligne de l'intégralité du texte.

*Constat général :* Grâce à l'injection structurée du contexte JSON et aux garde-fous explicites, le modèle n'a inventé aucun chiffre imaginaire, aucune $p$-value fallacieuse, ni aucune promesse de rentabilité de trading. Cependant, des imprécisions formelles sur la définition des tests et un artéfact lié au découpage des prompts ont été mis en évidence.

---

### 3. Matrice d'Audit Détaillée — Expérience 1 : Hypothèses Statistiques (EDA)

*Fichier source vérifié :* [`results/llm/experiments/response_1_hypothesis_generation.txt`](file:///c:/Users/kenza/Desktop/Financial_Time_Series_Restart/results/llm/experiments/response_1_hypothesis_generation.txt)

| Réf | Extrait de la Réponse du Modèle | Preuve Réelle (JSON / Projet) | Statut | Analyse Critique et Correction |
|:---:|---|---|:---:|---|
| **E1.1** | Tableau des tests ADF et KPSS avec les valeurs exactes : Oil stat $-1.8860$, $p=0.339$, KPSS $2.7765$ ; Gold stat $2.1901$, $p=0.999$, etc. | `stationarity_tests.csv` / `01_EDA.ipynb` (Cellule 27) | **SUPPORTED** | Concordance numérique absolue à 4 décimales. Aucune hallucination de valeur. |
| **E1.2** | *"Les deux tests s’accordent pour rejeter la stationnarité (ADF non significatif, KPSS significatif)."* | Définition des hypothèses nulles : ADF ($H_0$: racine unitaire), KPSS ($H_0$: stationnarité). | **OVERSIMPLIFIED** | **Raccourci de langage :** Le test ADF ne "rejette" pas la stationnarité ; il échoue à rejeter la présence d'une racine unitaire. Seul le test KPSS rejette formellement l'hypothèse nulle de stationnarité. |
| **E1.3** | *"Les rendements (log-différences) sont stationnaires : ADF très significatif, KPSS non significatif."* | Rendements : ADF $p < 10^{-20}$, KPSS $p \in [0.082, 0.10] > 0.05$. | **SUPPORTED** | Conclusion économétrique exacte : la différenciation première produit des séries $I(0)$. |
| **E1.4** | Tableau des tests ARCH-LM d'Engle : Oil stat $120.4019$, $p=4.19 \times 10^{-21}$ ; Gold stat $124.0399$ ; S&P 500 stat $148.8624$. | `arch_lm_training_returns.csv` | **SUPPORTED** | Reproduction strictement exacte des statistiques LM et de Fisher au retard 10. |
| **E1.5** | Formulation des hypothèses de modélisation : $H_1$ (moyenne nulle/constante), $H_2$ (autocorrélation résiduelle), $H_3$ (hétéroscédasticité conditionnelle), $V_1$ (GARCH), $V_2$ (asymétrie/levier), $V_3$ (DCC-GARCH multivarié). | Méthodologie standard Box-Jenkins et Bollerslev (1986). | **SUPPORTED** | Structuration séquentielle exemplaire (modélisation de la moyenne d'abord, de la variance ensuite). |

---

### 4. Matrice d'Audit Détaillée — Expérience 2 : Vulgarisation pour Décideur

*Fichier source vérifié :* [`results/llm/experiments/response_2_plain_language_explanation.txt`](file:///c:/Users/kenza/Desktop/Financial_Time_Series_Restart/results/llm/experiments/response_2_plain_language_explanation.txt)

| Réf | Extrait de la Réponse du Modèle | Preuve Réelle (JSON / Projet) | Statut | Analyse Critique et Correction |
|:---:|---|---|:---:|---|
| **E2.1** | *"Or : MAE ≈ 0,011 et RMSE ≈ 0,014 – légèrement inférieurs à la prévision naïve 'zéro rendement' (MAE = 0,01113... améliorations de -0,1% à -0,5%). Les modèles n'apportent pas d'avantage mesurable."* | `comparative_evaluation.csv` (Lignes 1-6 : MAE improvement $-0.13\%$, RMSE $-0.48\%$). | **SUPPORTED** | Parfaite compréhension : le modèle admet honnêtement que prédire zéro est plus précis qu'ARIMA/SARIMA/VAR sur l'Or. |
| **E2.2** | *"Pétrole : MAE entre 0,0287 et 0,0289... améliorations très modestes (≤ 0,8% de MAE, ≤ 0,7% de RMSE)... gains non significatifs."* | `comparative_evaluation.csv` (Lignes 7-12 : MAE improvement $+0.50\%$ ARIMA, $+0.09\%$ VAR). | **SUPPORTED** | Évaluation lucide : refus de qualifier ces gains infimes d'opportunité d'arbitrage. |
| **E2.3** | *"S&P 500 : Aucun modèle de prévision de rendement n’a été estimé pour le S&P 500 dans le jeu de résultats fourni."* | Le projet a estimé ARIMA, SARIMA et VAR sur le S&P 500 (`comparative_evaluation.csv`, Lignes 13-18). | **CONTRADICTED (Artéfact de Prompt)** | **Origine identifiée :** Dans `run_experiments.py`, la fonction `build_experiment_2_prompt` a tronqué la table à `comp[:12]`, omettant les lignes 12 à 17. Le LLM a été factuel vis-à-vis du snippet reçu, mais l'affirmation est fausse pour le projet global. |
| **E2.4** | Causalité de Granger S&P 500 $\to$ Or ($F=10.43, p=0.0013$) : *"Ce n'est PAS une preuve de 'cause réelle'... simple corrélation temporelle... ne pas baser une stratégie de trading sur ce seul signal."* | `var_granger_causality.csv` ($F=10.4288, p=0.00125$) et `METHODOLOGY_NOTES.md`. | **SUPPORTED** | Respect exemplaire de la mise en garde épistémologique demandée. |
| **E2.5** | Volatilité GARCH vs Moyenne : *"GARCH a un QLIKE meilleur (-7.394 vs -7.368 Or, -5.516 vs -4.837 Pétrole, -8.649 vs -8.511 S&P 500)... Les métriques portent sur la variance et ne sont pas comparables aux MAE/RMSE de rendements."* | `garch11_student_t_forecast_metrics.csv` | **SUPPORTED** | Séparation formelle stricte entre prévision directionnelle et modélisation du risque. |
| **E2.6** | Reprise fidèle des 5 notes méthodologiques : divergence Python/R sur Pétrole SARIMA (0.087 AIC), résidus VAR corrélés ($p=0.028$), contradiction de Johansen, DM ajusté Holm ($N=25$). | `METHODOLOGY_NOTES.md` et `BLOC_A_CONCLUSIONS.md`. | **SUPPORTED** | Assimilation parfaite des contraintes méthodologiques transmises dans le contexte. |

---

### 5. Matrice d'Audit Détaillée — Expérience 3 : Recommandations et Gestion des Risques

*Fichier source vérifié :* [`results/llm/experiments/response_3_risk_recommendations.txt`](file:///c:/Users/kenza/Desktop/Financial_Time_Series_Restart/results/llm/experiments/response_3_risk_recommendations.txt)

| Réf | Extrait de la Réponse du Modèle | Preuve Réelle (JSON / Projet) | Statut | Analyse Critique et Correction |
|:---:|---|---|:---:|---|
| **E3.1** | Tableau comparatif MSE RNN vs Baseline : Oil LSTM $0.0015735$ vs Base $0.0015520$ ; Gold LSTM $0.0002131$ vs Base $0.0002116$ ; S&P 500 LSTM $6.14 \times 10^{-5}$ vs Base $6.23 \times 10^{-5}$. | `final_test_metrics.csv` (Deep Learning multistep 5-pas). | **SUPPORTED** | Données numériques 100 % exactes. Démontre la transparence sur l'échec des réseaux récurrents. |
| **E3.2** | *"Directionnel : les prévisions de rendement des RNN ne sont pas plus fiables que le simple zero-forecast... ne doivent pas être utilisées pour des stratégies de trading."* | `dm_test_results_holm.csv` ($p_{\text{adj}} = 1.0$). | **SUPPORTED** | Conformité déontologique totale avec le mandat de gestion des risques. |
| **E3.3** | *"Procédure : Calculer le VaR_95 = z_0.95 * sigma_t... Limite = k * sigma_t (k ≈ 3)."* | Formule de la Value-at-Risk paramétrique sous hypothèse de distribution. | **OVERSIMPLIFIED** | **Nuance nécessaire :** La formule $z_\alpha \cdot \sigma_t$ suppose une loi normale ($z_{0.95} = 1.645$). Or le modèle retenu est un **GARCH-Student-t** avec $\nu \in [5.08, 8.71]$. Le quantile critique doit être extrait de la loi de Student normalisée. Aux degrés de liberté estimés, le quantile standardisé à 95 % vaut environ 1,57–1,62 : le $z = 1{,}645$ gaussien **surestime donc très légèrement** la VaR au seuil de 95 %. En revanche, aux seuils extrêmes (99 % et au-delà), le quantile de Student standardisé atteint environ 2,4–2,6 contre 2,33 en gaussien : la formule proposée **sous-estime alors le risque de queue**, et le facteur « $k \approx 3$ pour un niveau de 99 % » invoqué par le modèle ne correspond à aucune des deux lois (2,33 en gaussien, $\approx$ 2,5–3,0 en Student-t selon $\nu$). Le modèle mentionne toutefois judicieusement cette dépendance distributionnelle dans sa section limites. La Value-at-Risk n'a pas été implémentée dans ce projet : ce point porte uniquement sur une proposition du LLM. |
| **E3.4** | Limites méthodologiques répertoriées : non-normalité et queues lourdes, instabilité temporelle des paramètres GARCH, brièveté de la période de test ($N=126$, $N=25$ pour DM). | Diagnostic économétrique du projet. | **SUPPORTED** | Identification lucide des conditions de fragilité opérationnelle. |
| **E3.5** | Troncature finale à la ligne 54 : `| **Non‑prise en compte des corrélations inter‑actifs** |` sans texte de fin de tableau. | Paramètre `max_tokens = 2048` dans `run_experiments.py`. | **OVERSIMPLIFIED (Coupure de tokens)** | Le modèle a épuisé sa fenêtre de génération de 2 048 tokens avant d'avoir pu expliciter le scénario d'échec de la dernière ligne. Ce comportement illustre une contrainte technique bien connue des LLMs. |

---

### 6. Bilan et Recommandations pour la Soutenance

L'audit prouve que lorsque l'on fournit à un LLM un contexte JSON structuré et des contraintes d'intégrité explicites :
1. **Il ne fabrique pas de fausses performances :** L'illusion courante selon laquelle les réseaux de neurones ou l'ARIMA seraient "très prédictifs" a été entièrement neutralisée par les données factuelles.
2. **Il adopte une prudence professionnelle louable :** Aucun conseil financier imprudent n'a été formulé.
3. **Le rôle du Data Scientist reste indispensable :** Seul l'auditeur humain a pu déceler le raccourci sur la définition d'ADF (E1.2), le besoin d'ajuster le quantile de Student dans la VaR (E3.3), et l'artéfact de troncature du prompt sur le S&P 500 (E2.3).
