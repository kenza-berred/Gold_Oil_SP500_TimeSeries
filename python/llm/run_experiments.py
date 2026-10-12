"""
Bloc C — Script d'exécution des expériences d'intégration LLM.
Lit les résultats réels depuis 'results/llm/financial_project_context.json',
construit les trois prompts spécialisés et appelle l'API Groq (si configurée).

Sécurité et intégrité :
- Ne hardcode aucun secret.
- Lit les identifiants exclusivement depuis os.environ.get("GROQ_API_KEY").
- Si la clé est absente, consigne le statut 'PENDING_API_KEY' sans inventer de réponse.
- Préserve les réponses déjà validées et permet de relancer une seule expérience (--experiment N).
- Gère les limites de débit HTTP 429 en respectant le header 'Retry-After' sans boucle infinie.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
CONTEXT_PATH = ROOT / "results" / "llm" / "financial_project_context.json"
EXPERIMENTS_DIR = ROOT / "results" / "llm" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def load_context():
    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(f"Fichier de contexte introuvable : {CONTEXT_PATH}")
    with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_experiment_1_prompt(ctx):
    """
    Expérience 1 : Génération d'hypothèses statistiques à partir de l'EDA.
    Fournit les statistiques descriptives, tests ADF/KPSS et diagnostics ARCH.
    """
    proj = ctx.get("project", {})
    results = ctx.get("results", {})
    stationarity = results.get("stationarity_tests", {}).get("data", [])
    arch = results.get("arch_lm", {}).get("data", [])
    constraints = ctx.get("interpretation_constraints", [])

    eda_summary = {
        "actifs": proj.get("assets"),
        "observations_apprentissage": proj.get("train_observations"),
        "periode": "Octobre 2021 à Avril 2026 (90% train)",
        "tests_stationnarite": stationarity,
        "tests_arch_lm": arch,
    }

    prompt = f"""Tu es un professeur d'économétrie et de science des données financières.
Voici les résultats empiriques réels issus de la phase d'analyse exploratoire (EDA) d'un projet universitaire sur l'Or (Gold), le Pétrole brut WTI (Oil) et le S&P 500 :

{json.dumps(eda_summary, indent=2, ensure_ascii=False)}

CONTRAINTES STRICTES DE RIGUEUR :
{json.dumps(constraints[:4], indent=2, ensure_ascii=False)}

CONSIGNE :
1. Analyse les propriétés de stationnarité (ADF vs KPSS sur prix et rendements) et formule des hypothèses testables sur la structure temporelle des rendements.
2. Interprète les statistiques ARCH-LM obtenues (p-values < 10^-20) et formule des hypothèses sur la modélisation de la variance conditionnelle.
3. Propose une méthodologie économétrique séquentielle pour tester ces hypothèses (modèles de moyenne conditionnelle puis modèles de variance conditionnelle).
4. Distingue rigoureusement ce qui est déjà prouvé par les données fournies de ce qui reste une hypothèse à vérifier. N'invente aucun chiffre ni résultat futur."""

    return prompt.strip()


def build_experiment_2_prompt(ctx):
    """
    Expérience 2 : Vulgarisation pour un décideur non-technique.
    Fournit les performances des modèles ARIMA, SARIMA, VAR, GARCH et la causalité de Granger.
    """
    results = ctx.get("results", {})
    comp = results.get("comparative_evaluation", {}).get("data", [])
    granger = results.get("granger_causality", {}).get("data", [])
    garch = results.get("garch_forecast_metrics", {}).get("data", [])
    constraints = ctx.get("interpretation_constraints", [])

    model_summary = {
        "evaluation_comparative_rendements_face_au_zero": comp[:12],
        "test_causalite_granger_var1": granger,
        "evaluation_prevision_volatilite_garch": garch,
    }

    prompt = f"""Tu es un consultant en science des données s'adressant à un Directeur Financier non-statisticien.
Voici les résultats réels et vérifiés des modèles testés sur 126 jours de test (Avril à Octobre 2026) :

{json.dumps(model_summary, indent=2, ensure_ascii=False)}

RÈGLES ET CONTRAINTES D'INTÉGRITÉ SCIENTIFIQUE :
{json.dumps(constraints, indent=2, ensure_ascii=False)}

CONSIGNE :
1. Explique dans un langage d'affaires clair et accessible ce que révèlent ces modèles sur la prédictibilité des cours de l'Or, du Pétrole et du S&P 500.
2. Explique pourquoi les modèles complexes (ARIMA, SARIMA, VAR) ne parviennent pas à battre de manière significative une prévision naïve de "zéro rendement" (lien avec l'efficience des marchés financiers).
3. Explique le résultat du test de Granger (S&P 500 -> Gold avec p = 0.00125) sans commettre l'erreur d'affirmer qu'il s'agit d'une causalité physique réelle.
4. Distingue la prévision de la direction des cours (très faible) de la prévision du risque / volatilité par GARCH (très performante). N'invente aucune métrique ni affirmation non étayée."""

    return prompt.strip()


def build_experiment_3_prompt(ctx):
    """
    Expérience 3 : Recommandations simulées de gestion des risques et limites explicites.
    Fournit les métriques GARCH, Deep Learning et Diebold-Mariano.
    """
    results = ctx.get("results", {})
    dl = results.get("deep_learning_metrics", {}).get("data", [])
    garch = results.get("garch_forecast_metrics", {}).get("data", [])
    dm = results.get("dm_holm_tests", {}).get("data", [])
    constraints = ctx.get("interpretation_constraints", [])

    risk_data = {
        "garch_volatility_metrics": garch,
        "deep_learning_multistep_metrics": dl,
        "diebold_mariano_significance_tests": dm[:10],
    }

    prompt = f"""Tu es un analyste en gestion des risques quantitatifs au sein d'un comité d'investissement.
Voici les données chiffrées réelles obtenues par le projet de séries temporelles :

{json.dumps(risk_data, indent=2, ensure_ascii=False)}

CONTRAINTES CRITIQUES DE RESPONSABILITÉ :
- Tu NE DOIS PAS formuler de conseil en investissement personnalisé.
- Tu NE DOIS PAS affirmer que les modèles permettent de réaliser des gains financiers prévisibles ou réguliers.
- Tu DOIS distinguer explicitement la prévision directionnelle (non fiable) de la mesure du risque de volatilité (GARCH).
- Tu DOIS mentionner que les performances passées ne préjugent pas des performances futures.
- Tu DOIS souligner les limites des modèles (tests Diebold-Mariano non significatifs, non-normalité, instabilité).

CONSIGNE :
1. Rédige une note synthétique pour le comité des risques expliquant comment utiliser ces résultats concrètement (allocation de capital, dimensionnement des limites de risque, Value-at-Risk).
2. Explique pourquoi les réseaux de neurones (LSTM, GRU) n'apportent aucun avantage pour prédire les rendements quotidiens face au benchmark zéro.
3. Formule explicitement au moins 4 limites majeures et scénarios d'échec de ces modélisations."""

    return prompt.strip()


def call_groq_api(prompt, api_key, model=DEFAULT_MODEL, max_retries=1):
    """
    Appelle l'API REST de Groq via urllib natif sans dépendance externe.
    Gère les codes HTTP 429 en analysant Retry-After, avec au maximum 1 nouvelle tentative.
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Tu es un assistant scientifique expert en économétrie des séries temporelles et gestion des risques financiers. Tu respectes strictement les faits fournis sans jamais inventer de données.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "FinancialTimeSeries-BlocC/1.0",
        },
        method="POST",
    )

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after_hdr = exc.headers.get("Retry-After")
                wait_sec = 6.0
                if retry_after_hdr:
                    try:
                        wait_sec = float(retry_after_hdr)
                    except ValueError:
                        wait_sec = 6.0

                if attempt < max_retries and wait_sec <= 30.0:
                    print(f"  [!] Limite de débit atteinte (HTTP 429). Pause de {wait_sec:.1f}s selon Retry-After...")
                    time.sleep(wait_sec)
                    continue
                else:
                    raise RuntimeError(
                        f"HTTP Error 429: Too Many Requests (Limite de débit atteinte. Réessayez dans {wait_sec:.0f}s)"
                    ) from None
            raise


def load_existing_manifest():
    manifest_file = EXPERIMENTS_DIR / "experiments_manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def compute_overall_status(experiment_records, has_api_key):
    statuses = [e.get("status") for e in experiment_records]
    if not has_api_key:
        return "PENDING_API_KEY"
    if all(s == "COMPLETED" for s in statuses):
        return "COMPLETED"
    if any(s == "COMPLETED" for s in statuses) and any(s != "COMPLETED" for s in statuses):
        return "PARTIAL_SUCCESS"
    if all(s == "API_ERROR" for s in statuses):
        return "API_ERROR"
    return "IN_PROGRESS"


def main():
    parser = argparse.ArgumentParser(
        description="Bloc C — Exécuteur d'expériences LLM (Groq API)"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=int,
        choices=[1, 2, 3],
        help="ID de l'expérience unique à exécuter (1, 2 ou 3). Si omis, exécute les 3 expériences.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Identifiant du modèle Groq (défaut : {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BLOC C — EXÉCUTEUR D'EXPÉRIENCES LLM")
    print("=" * 60)

    ctx = load_context()
    print(f"Contexte chargé : {CONTEXT_PATH.name} ({len(ctx.get('results', {}))} jeux de données)")

    all_experiments = [
        {
            "id": 1,
            "name": "hypothesis_generation",
            "title": "Expérience 1 : Génération d'hypothèses statistiques (EDA)",
            "prompt_func": build_experiment_1_prompt,
        },
        {
            "id": 2,
            "name": "plain_language_explanation",
            "title": "Expérience 2 : Vulgarisation pour décideur non-technicien",
            "prompt_func": build_experiment_2_prompt,
        },
        {
            "id": 3,
            "name": "risk_recommendations",
            "title": "Expérience 3 : Recommandations simulées et gestion des risques",
            "prompt_func": build_experiment_3_prompt,
        },
    ]

    if args.experiment:
        experiments_to_run = [e for e in all_experiments if e["id"] == args.experiment]
        print(f"[CIBLAGE] Exécution ciblée : Expérience {args.experiment} uniquement.")
    else:
        experiments_to_run = all_experiments
        print("[CIBLAGE] Exécution globale des 3 expériences.")

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    has_api_key = bool(api_key)

    if has_api_key:
        print(f"[INFO] Clé API Groq détectée dans l'environnement. Modèle sélectionné : {args.model}")
    else:
        print("[AVERTISSEMENT] Aucune clé GROQ_API_KEY trouvée dans les variables d'environnement.")
        print("[INFO] Les gabarits de prompts et métadonnées seront générés avec le statut 'PENDING_API_KEY'.")

    # Charger le manifeste existant pour préserver les exécutions antérieures
    existing_manifest = load_existing_manifest()
    existing_experiments_map = {}
    if existing_manifest and "experiments" in existing_manifest:
        for exp in existing_manifest["experiments"]:
            existing_experiments_map[exp.get("id")] = exp

    run_records = {}

    for exp in experiments_to_run:
        exp_id = exp["id"]
        exp_name = exp["name"]
        print(f"\n--- Traitement de l'{exp['title']} ---")

        prompt_text = exp["prompt_func"](ctx)
        prompt_file = EXPERIMENTS_DIR / f"prompt_{exp_id}_{exp_name}.txt"
        response_file = EXPERIMENTS_DIR / f"response_{exp_id}_{exp_name}.txt"

        prompt_file.write_text(prompt_text, encoding="utf-8")
        print(f"  [+] Prompt sauvegardé : {prompt_file.name} ({len(prompt_text)} caractères)")

        if has_api_key:
            print(f"  [*] Envoi de la requête à Groq ({args.model})...")
            try:
                raw_response = call_groq_api(prompt_text, api_key, model=args.model)
                response_file.write_text(raw_response, encoding="utf-8")
                print(f"  [+] Réponse reçue et sauvegardée : {response_file.name} ({len(raw_response)} caractères)")
                exp_status = "COMPLETED"
                error_msg = None
            except Exception as exc:
                print(f"  [!] Erreur lors de l'appel API : {exc}")
                exp_status = "API_ERROR"
                error_msg = str(exc)
                response_file.write_text(f"ERREUR_APPEL_API: {exc}", encoding="utf-8")
        else:
            pending_notice = (
                f"STATUT : PENDING_API_KEY\n"
                f"Date : {datetime.now(timezone.utc).isoformat()}\n"
                f"Ce fichier contiendra la réponse brute et non éditée dès configuration de GROQ_API_KEY.\n"
                f"Le prompt exact est préparé et vérifié dans : {prompt_file.name}\n"
            )
            response_file.write_text(pending_notice, encoding="utf-8")
            print(f"  [i] Fichier de réponse initialisé (statut PENDING_API_KEY) : {response_file.name}")
            exp_status = "PENDING_API_KEY"
            error_msg = "GROQ_API_KEY non configurée dans l'environnement."

        run_records[exp_id] = {
            "id": exp_id,
            "name": exp_name,
            "status": exp_status,
            "prompt_file": prompt_file.name,
            "response_file": response_file.name,
            "error": error_msg,
        }

    # Fusionner les résultats avec l'existant en préservant les expériences non rejouées
    final_experiments_list = []
    for exp_def in all_experiments:
        eid = exp_def["id"]
        if eid in run_records:
            final_experiments_list.append(run_records[eid])
        elif eid in existing_experiments_map:
            final_experiments_list.append(existing_experiments_map[eid])
        else:
            final_experiments_list.append({
                "id": eid,
                "name": exp_def["name"],
                "status": "NOT_RUN",
                "prompt_file": f"prompt_{eid}_{exp_def['name']}.txt",
                "response_file": f"response_{eid}_{exp_def['name']}.txt",
                "error": None,
            })

    overall_status = compute_overall_status(final_experiments_list, has_api_key)

    manifest = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "Groq" if has_api_key else "None (Pending Key)",
        "model": args.model if has_api_key else "None",
        "status": overall_status,
        "experiments": final_experiments_list,
    }

    manifest_file = EXPERIMENTS_DIR / "experiments_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SUCCÈS] Manifeste d'expériences mis à jour : {manifest_file.name}")
    print(f"Statut global d'exécution : {overall_status}")


if __name__ == "__main__":
    main()
