# Bloc C — Interprétation Humaine de Référence (Benchmark Étudiant)
## Analyse Économétrique et Statistique des Résultats du Projet

**Auteur :** Étudiante en Ingénierie Data Science & IA (TEK-UP)  
**Date :** 12 Octobre 2026  
**Fichier source de données :** [`results/llm/financial_project_context.json`](file:///c:/Users/kenza/Desktop/Financial_Time_Series_Restart/results/llm/financial_project_context.json)  
**Rôle méthodologique :** Ce document constitue l'interprétation humaine experte rédigée **avant** toute confrontation avec les modèles de langage génératifs (LLM). Il sert de référence de vérité terrain (*ground truth*) pour auditer les réponses de l'IA et détecter d'éventuelles hallucinations statistiques.

---

### 1. Propriétés Temporelles et Stationnarité des Séries

L'analyse des séries de prix de clôture quotidiens de l'Or (Gold), du Pétrole brut WTI (Oil) et de l'indice S&P 500 sur la période d'octobre 2021 à octobre 2026 (1 255 observations synchronisées) met en évidence les faits stylisés suivants :

1. **Non-stationnarité des prix en niveau ($I(1)$) :**
   - Les tests de Dickey–Fuller Augmenté (ADF) conduits sur les prix bruts ne permettent pas de rejeter l'hypothèse nulle de racine unitaire au seuil de 5 % ($p$-values respectives : 0,812 pour l'Or, 0,404 pour le Pétrole, 0,723 pour le S&P 500).
   - Les tests KPSS rejettent catégoriquement la stationnarité en niveau ($p$-values $< 0,01$ pour les trois actifs).
   - *Conclusion :* Les trajectoires des prix suivent des processus intégrés d'ordre 1 ($I(1)$), dominés par des tendances stochastiques (marches aléatoires avec dérive).

2. **Stationnarité stricte des rendements logarithmiques ($I(0)$) :**
   - Pour les log-rendements quotidiens $r_t = \ln(P_t / P_{t-1})$, le test ADF rejette massivement la présence d'une racine unitaire ($p$-value $< 10^{-15}$ pour les trois actifs, statistiques ADF $\le -33,4$).
   - Le test KPSS échoue à rejeter la stationnarité ($p$-value $> 0,10$).
   - *Conclusion :* La différenciation logarithmique d'ordre 1 stationnarise parfaitement les séries, rendant les modèles univariés et multivariés applicables sur les rendements sans risque de régression fallacieuse (*spurious regression*).

---

### 2. Modélisation de la Moyenne Conditionnelle : ARIMA, SARIMA et VAR

La modélisation de la moyenne conditionnelle vise à capturer d'éventuelles dépendances linéaires passées pour prédire les rendements futurs.

1. **Sélection et adéquation des modèles ARIMA :**
   - Le critère d'information d'Akaike (AIC) sur les 1 128 observations d'entraînement a sélectionné :
     - **Or :** $\text{ARIMA}(0,0,1)$ ($\text{AIC} = -6960,67$), révélant une très légère mémoire à court terme de type moyenne mobile ($q=1$).
     - **Pétrole :** $\text{ARIMA}(2,0,2)$ ($\text{AIC} = -5175,57$), reflétant une dynamique autorégressive et moyenne mobile d'ordre 2.
     - **S&P 500 :** $\text{ARIMA}(0,0,0)$ ($\text{AIC} = -8101,60$), indiquant l'absence absolue de structure autorégressive linéaire au-delà d'une constante (conformité avec l'hypothèse des marchés efficients en forme faible).
   - **Blanchiment des résidus :** Les résidus passent avec succès le test de Ljung–Box au retard 10 et 20 sous Python (absence d'autocorrélation linéaire résiduelle). La seule nuance est observée sous R pour le Pétrole ($p = 0,034$ au retard 10), causée par une correction plus sévère des degrés de liberté ($\text{df} = 10 - (2+2) = 6$).

2. **Extension saisonnière SARIMA ($S=5$) :**
   - L'introduction d'une composante hebdomadaire (période de négociation boursière $S=5$ jours) n'apporte aucun gain pour l'Or et le S&P 500 (sélection de $(0,0,0)_5$).
   - Pour le Pétrole, une divergence minime apparaît entre Python (sélection de $(1,0,0)_5$) et R (sélection de $(0,0,1)_5$). Cette divergence provient d'un écart d'AIC infinitésimal de **0,087 point**, lié à des conventions d'optimisation numérique entre librairies et non à une rupture méthodologique.

3. **Modélisation multivariée VAR(1) :**
   - L'estimation conjointe du système [Oil, Gold, S&P 500] retient l'ordre $p=1$.
   - Les résidus du VAR(1) affichent toutefois une légère autocorrélation résiduelle selon le test multivarié de Portmanteau ($p = 0,028$ sous Python, $p = 0,031$ sous R), indiquant que le modèle VAR(1) n'épuise pas totalement l'interdépendance linéaire fine entre actifs.

4. **Performance prédictive face au Baseline Zéro-Rendement :**
   - Sur les 126 jours de l'échantillon test, les modèles ARIMA, SARIMA et VAR sont comparés au benchmark naïf de zéro-rendement ($\hat{r}_{t+h} = 0$).
   - **Constat fondamental :** Les gains prédictifs sont **économiquement et statistiquement négligeables** :
     - Pour l'Or, les modèles font **systématiquement pire** que de prédire zéro ($\text{MAE}$ dégradée de $-0,13\%$ à $-0,14\%$, $\text{RMSE}$ dégradée de $-0,48\%$ à $-0,51\%$).
     - Pour le Pétrole, les améliorations sont marginales ($+0,50\%$ en MAE pour ARIMA, $+0,09\%$ pour VAR).
     - Pour le S&P 500, les gains oscillent entre $+0,18\%$ et $+0,23\%$ en MAE.
   - *Interprétation data science :* Les log-rendements quotidiens sur les marchés liquides sont dominés par le bruit blanc. Les modèles linéaires ne fournissent aucun avantage exploitable pour le trading directionnel.

---

### 3. Causalité au Sens de Granger : Rigueur Économétrique vs Illusion Causale

Le test de causalité de Granger dans le système VAR(1) teste si les retards d'une variable $X$ réduisent significativement la variance de l'erreur de prévision d'une variable $Y$, au-delà de l'information déjà contenue dans les retards de $Y$.

- **Résultats vérifiés des 6 paires :**
  - Or $\to$ Pétrole : $p = 0,687$ (non significatif)
  - S&P 500 $\to$ Pétrole : $p = 0,492$ (non significatif)
  - Pétrole $\to$ Or : $p = 0,471$ (non significatif)
  - **S&P 500 $\to$ Or : $F = 10,43$, $p = 0,00125$ (hautement significatif)**
  - Pétrole $\to$ S&P 500 : $p = 0,669$ (non significatif)
  - Or $\to$ S&P 500 : $p = 0,664$ (non significatif)

- **Mise en garde épistémologique cruciale :**
  La relation S&P 500 $\to$ Or indique une **prédictibilité linéaire incrémentale retardée** (associée empiriquement au rôle de l'Or comme valeur refuge réagissant avec un décalage d'un jour aux secousses des marchés actions). En aucun cas cette statistique ne constitue une preuve de causalité mécanique, physique ou structurelle dans le monde réel. Prétendre que "les actions font bouger l'or par causalité directe" serait une erreur conceptuelle majeure.

---

### 4. Hétéroscédasticité Conditionnelle et Volatilité GARCH(1,1)

Bien que les moyennes des rendements soient quasi-imprévisibles, la variance conditionnelle des rendements exhibe une structure hautement persistante :

1. **Preuve irréfutable d'effets ARCH :**
   - Le test ARCH-LM d'Engle au retard 10 rejette l'homoscédasticité avec des $p$-values inférieures à $10^{-20}$ sur les trois actifs (statistique LM : 120,40 pour le Pétrole, 124,04 pour l'Or, 148,86 pour le S&P 500).
   - Les rendements financiers violent l'hypothèse de variance constante : les fortes amplitudes passées génèrent de fortes amplitudes futures (*volatility clustering*).

2. **Ajustement GARCH(1,1) avec innovations Student-t :**
   - La loi de Student-t s'impose face à la loi normale car elle absorbe l'excès d'aplatissement (*kurtosis*) et les queues de distribution épaisses :
     - Degrés de liberté $\nu$ : 5,08 (Or), 6,79 (S&P 500), 8,71 (Pétrole).
     - Persistance de la volatilité $\alpha_1 + \beta_1$ : 0,9688 (Pétrole), 0,9831 (Or), 0,9864 (S&P 500).
   - Cette persistance très proche de 1 démontre que les chocs de volatilité mettent plusieurs semaines à se résorber.
   - Les résidus standardisés $\hat{z}_t = \hat{\epsilon}_t / \hat{\sigma}_t$ sont parfaitement blanchis au second ordre (Ljung–Box sur les carrés : $p > 0,90$).

3. **Prévision de volatilité hors-échantillon :**
   - Évalué via la perte de quasi-vraisemblance (QLIKE) sur les 126 jours de test face à un modèle de variance historique constante :
     - Or : QLIKE GARCH $= -7,394$ vs Baseline $= -7,368$ (GARCH supérieur).
     - Pétrole : QLIKE GARCH $= -5,516$ vs Baseline $= -4,837$ (GARCH supérieur).
     - S&P 500 : QLIKE GARCH $= -8,649$ vs Baseline $= -8,511$ (GARCH supérieur).
   - *Séparation fondamentale :* Ces métriques évaluent la capacité à anticiper le **risque** ($\sigma_t^2$), et non la **direction** du prix ($r_t$). Elles ne doivent jamais être confondues avec les métriques MAE/RMSE de prévision des rendements.

---

### 5. Réseaux de Neurones Récurrents (LSTM et GRU)

1. **Protocole rigoureux :**
   - Découpage chronologique strict, normalisation `StandardScaler` calibrée exclusivement sur les fenêtres d'apprentissage (absence totale de fuite temporelle), validation croisée temporelle (*walk-forward*) en 3 plis extensifs.
   - Prévision multi-pas sur un horizon de $h = 5$ jours ouvrés (25 blocs non-chevauchants, soit 125 prévisions).

2. **Résultats empiriques :**
   - **LSTM :** RMSE Pétrole $-0,69\%$ (pire que baseline), RMSE Or $-0,35\%$ (pire que baseline), RMSE S&P 500 $+0,70\%$ (gain minime).
   - **GRU :** RMSE Pétrole $-0,37\%$ (pire), RMSE Or $-0,37\%$ (pire), RMSE S&P 500 $-0,29\%$ (pire).
   - *Bilan scientifique :* Les réseaux profonds récurrents souffrent d'un surapprentissage structurel face au bruit financier. Sur un horizon de rendement quotidien, ils ne parviennent pas à surpasser la prévision nulle. Leur complexité architecturale et leur coût computationnel ne sont pas justifiés pour le point-forecasting univarié/multivarié des rendements.

---

### 6. Tests de Supériorité Prédictive de Diebold–Mariano

1. **Procédure statistique :**
   - 30 tests bilatéraux de Diebold–Mariano avec correction pour petits échantillons de Harvey–Leybourne–Newbold (HLN) et contrôle conservateur du taux d'erreur par famille de Holm–Bonferroni.
2. **Constat statistique :**
   - Après ajustement de Holm, **aucune comparaison n'est statistiquement significative à 5 %** ($p_{\text{adj}} = 1{,}000$ pour la comparaison la plus favorable, p brute minimale 0,04825).
   - Aucun modèle (ARIMA, SARIMA, VAR, LSTM, GRU) ne peut être déclaré statistiquement supérieur au benchmark zéro-rendement.
3. **Limite d'échantillonnage documentée :**
   - L'alignement multi-modèles dans `diebold_mariano_test.py` ayant filtré sur le premier pas du bloc ($h=1$), la taille d'échantillon effective a été réduite à $N=25$. Bien que statistiquement valide pour éviter la corrélation sérielle des erreurs de prévision non-chevauchantes, cette taille d'échantillon restreinte diminue la puissance du test (risque d'erreur de type II). Un non-rejet ne doit donc pas être lu comme une preuve d'équivalence. Par ailleurs, les origines de prévision diffèrent : ARIMA/SARIMA projetés en origine fixe depuis la fin de l'apprentissage, LSTM/GRU recalculés en origine glissante à chaque début de bloc — la comparaison porte sur des dates cibles appariées, non sur un protocole d'origine strictement identique.

---

### 7. Analyse de Cointégration et Décision VECM

1. **Tests de Johansen sur les prix logarithmiques :**
   - Test de la trace à 5 % : indique un rang de cointégration $r = 3$ (plein rang).
   - Test de la valeur propre maximale à 5 % : indique un rang $r = 1$.
   - Sous R (`urca`), le test de la trace échoue marginalement à rejeter $r = 0$ (statistique 34,80 pour une valeur critique de 34,91) tandis que la valeur propre maximale indique $r = 1$.
2. **Dilemme économétrique et décision rigoureuse :**
   - Les deux statistiques de Johansen se contredisent formellement ($\text{tests\_agree} = \text{False}$).
   - Un rang $r = 3$ dans un système à 3 variables impliquerait que les trois séries de prix sont stationnaires en niveau, ce qui contredit frontalement les tests ADF et KPSS ($I(1)$).
   - La décision scientifique actée dans `python/vecm_model.py` et consignée dans `results/python/vecm/vecm_decision.csv` a été de **ne pas estimer de modèle VECM forcé** (`vecm_estimated: False`).
   - Forcer un rang arbitraire aurait constitué une grave entorse méthodologique. Le VECM relevant du bloc « approfondissement et extensions » du barème, son estimation n'a pas été poursuivie : le projet ne présente donc aucun modèle VECM, aucune prévision ni aucune métrique VECM.

---

### 8. Implications Pratiques et Limites Managériales

1. **Inexistence d'un "Alpha" directionnel :**
   Aucun modèle du projet ne permet d'envisager une stratégie de trading algorithmique rentable basée sur la prévision directionnelle des cours. Les marchés de l'Or, du Pétrole et du S&P 500 intègrent l'information quasi-instantanément.
2. **Valeur réelle pour la gestion des risques :**
   L'apport opérationnel du projet réside intégralement dans la modélisation de la volatilité par **GARCH(1,1)-Student-t**. Elle permet de calibrer des enveloppes de risque, d'alimenter un dimensionnement de capital économique (Value-at-Risk) et d'anticiper les phases de turbulences de marché. La VaR paramétrique n'a toutefois pas été implémentée ni chiffrée dans ce projet : son calcul à partir de $\hat{\sigma}_t$ et du quantile de Student normalisé reste une extension recommandée.
3. **Incertitude et fragilité des inférences :**
   Toute recommandation opérationnelle doit être assortie d'intervalles de confiance larges, compte tenu de la non-normalité des résidus et de l'instabilité potentielle des régimes de marché.
