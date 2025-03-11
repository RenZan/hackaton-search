import requests
import json
import time
import re
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from functools import partial
from datetime import datetime

# Configuration du logging
DEBUG = False
log_level = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configuration des URLs des APIs
SEARXNG_API_URL = "http://10.13.0.5:8081/search"
CRAWL4AI_API_URL = "http://10.13.0.5:11235/crawl"
CRAWL4AI_TASK_URL = "http://10.13.0.5:11235/task"
OPENAI_ENDPOINT = "http://192.168.1.11:11434/v1/completions"

# Token d'authentification pour Crawl4AI
CRAWL4AI_API_TOKEN = "toto"

# Limites de concurrence
CONCURRENCY_LIMIT = 2

# Date actuelle
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")

# Variable pour signaler l'arrêt
shutdown_flag = False

# Gestionnaire de signal pour Ctrl+C
def signal_handler(sig, frame):
    global shutdown_flag
    logger.info("Interruption détectée (Ctrl+C), arrêt en cours...")
    shutdown_flag = True
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Configuration des retries pour les requêtes
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))

# Fonction pour nettoyer le Markdown
def clean_markdown(md):
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"^\s*-+\s*$", "", md, flags=re.MULTILINE)
    md = re.sub(r"https?://\S+", "", md)
    md = re.sub(r"\[.*?\]\(.*?\)", "", md)
    return md.strip()

# Fonction pour découper le texte en chunks
def chunk_text(text, chunk_size=8000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            break_index = max(text.rfind('\n', start, end), text.rfind(' ', start, end))
            if break_index > start:
                end = break_index
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# Fonction pour obtenir les résultats de recherche via SearXNG
def get_search_results(query, max_results=3):
    if shutdown_flag:
        return []
    logger.info(f"[SEARXNG] Envoi de la requête : '{query}'")
    params = {
        "q": query,
        "format": "json",
        "time_range": "year"
    }
    try:
        response = session.get(SEARXNG_API_URL, params=params, timeout=15)
        response.raise_for_status()
        results = response.json().get("results", [])
        urls = [result["url"] for result in results[:max_results]]
        logger.info(f"[SEARXNG] URLs récupérées : {len(urls)}")
        return urls
    except requests.RequestException as e:
        logger.error(f"[SEARXNG] Erreur lors de la requête : {e}")
        return []

# Fonction pour extraire le contenu d'une page via Crawl4AI
def extract_content(url, start_time=None, time_limit=None):
    if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
        return ""
    logger.info(f"[CRAWL4AI] Lancement du crawl pour {url}")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {CRAWL4AI_API_TOKEN}"}
    payload = {
        "urls": url,
        "priority": 10,
        "options": {"ignore_links": True, "ignore_images": True, "escape_html": True, "body_width": 80}
    }
    try:
        response = session.post(CRAWL4AI_API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        task_id = response.json().get("task_id")
        if not task_id:
            logger.error(f"[CRAWL4AI] Aucun task_id retourné pour {url}")
            return ""
        task_start_time = time.time()
        while time.time() - task_start_time < 20:
            if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
                logger.info(f"[CRAWL4AI] Arrêt du crawl pour {url} suite à interruption ou limite de temps")
                return ""
            task_response = session.get(f"{CRAWL4AI_TASK_URL}/{task_id}", headers=headers, timeout=5)
            task_result = task_response.json()
            if task_result.get("status") == "completed":
                result = task_result.get("result", {})
                content = result.get("fit_markdown", "") or result.get("markdown", "")
                if not isinstance(content, str):
                    logger.warning(f"[CRAWL4AI] Contenu invalide pour {url}, retour à une chaîne vide")
                    content = ""
                cleaned_content = clean_markdown(content)
                logger.info(f"[CRAWL4AI] Contenu extrait (longueur={len(cleaned_content)}) pour {url}")
                return cleaned_content
            time.sleep(2)
        logger.error(f"[CRAWL4AI] Timeout dépassé pour {url}")
        return ""
    except requests.RequestException as e:
        logger.error(f"[CRAWL4AI] Erreur lors de l'extraction pour {url} : {e}")
        return ""

# Fonction pour analyser un chunk avec le LLM
def analyze_chunk(chunk, query, initial_query, start_time=None, time_limit=None, max_retries=0):
    if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
        return []
    logger.info(f"[LLM] Analyse d'un chunk de longueur {len(chunk)} pour '{query}'")
    prompt = (
        f"Tu es un expert en recherche. La date actuelle est {CURRENT_DATE}. Analyse ce texte et extrais jusqu'à 3 faits ou données factuelles clés "
        f"en rapport direct avec le sujet initial '{initial_query}'. Chaque fait doit être concis, précis "
        f"(incluant noms, chiffres ou dates), pertinent à la date actuelle ({CURRENT_DATE}), et unique. Ignore les informations hors sujet ou obsolètes.\n\n"
        f"Retourne chaque fait sous forme de liste à puces (exemple : '- Fait 1').\n\n"
        f"Texte :\n{chunk}"
    )
    payload = {"prompt": prompt, "max_tokens": 1500, "temperature": 0.5}
    headers = {"Content-Type": "application/json"}
    for attempt in range(max_retries + 1):
        if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
            return []
        try:
            response = session.post(OPENAI_ENDPOINT, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            text = response.json().get("choices", [{}])[0].get("text", "").strip()
            facts = re.findall(r"- (.+)", text)
            if facts:
                logger.debug(f"[LLM] Faits extraits : {facts}")
                return facts
            else:
                logger.warning(f"[LLM] Aucun fait pertinent extrait (tentative {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    time.sleep(1)
                continue
        except requests.RequestException as e:
            logger.warning(f"[LLM] Erreur d'analyse (tentative {attempt + 1}/{max_retries + 1}) : {e}")
            if attempt < max_retries:
                time.sleep(1)
            continue
    logger.error("[LLM] Échec après plusieurs tentatives, aucun fait extrait.")
    return []

# Fonction pour analyser le contenu complet
def analyze_content(content, query, initial_query, start_time=None, time_limit=None):
    if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
        return []
    chunks = chunk_text(content, chunk_size=8000)
    logger.info(f"[LLM] Contenu découpé en {len(chunks)} chunk(s)")
    with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
        analyze_with_time = partial(analyze_chunk, query=query, initial_query=initial_query, start_time=start_time, time_limit=time_limit)
        learnings_chunks = list(executor.map(analyze_with_time, chunks))
    all_learnings = [fact for sublist in learnings_chunks for fact in sublist if fact]
    unique_learnings = list(dict.fromkeys(all_learnings))
    logger.info(f"[LLM] Faits uniques extraits : {len(unique_learnings)}")
    return unique_learnings

# Nouvelle fonction pour valider et générer des requêtes de suivi
def validate_facts(current_knowledge, initial_query, start_time=None, time_limit=None, max_retries=3):
    if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
        return []
    logger.info("[LLM] Validation des faits accumulés pour générer des requêtes de suivi")
    all_learnings = [fact for learnings in current_knowledge.values() for fact in learnings]
    prompt = (
        f"Tu es un expert en recherche. La date actuelle est {CURRENT_DATE}. Analyse les faits suivants liés à '{initial_query}' et identifie "
        f"jusqu'à 3 faits qui nécessitent une vérification ou un approfondissement (ex. données incertaines, chiffres à confirmer, manque de précision). "
        f"Pour chaque fait sélectionné, propose une question de recherche concise et spécifique pour le vérifier ou le préciser, pertinente à {CURRENT_DATE}. "
        f"Retourne les résultats sous forme de liste à puces avec le fait suivi de la question (exemple : '- Fait : ... - Question : ...').\n\n"
        f"Faits disponibles :\n" + "\n".join([f"- {fact}" for fact in all_learnings]) + "\n\n"
    )
    payload = {"prompt": prompt, "max_tokens": 1500, "temperature": 0.5}
    headers = {"Content-Type": "application/json"}
    for attempt in range(max_retries + 1):
        if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
            return []
        try:
            response = session.post(OPENAI_ENDPOINT, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            text = response.json().get("choices", [{}])[0].get("text", "").strip()
            items = re.findall(r"- Fait : (.+?)\s*- Question : (.+)", text)
            follow_up_queries = [question for _, question in items]
            logger.info(f"[LLM] Requêtes de suivi générées : {follow_up_queries}")
            return follow_up_queries
        except requests.RequestException as e:
            logger.warning(f"[LLM] Erreur lors de la validation (tentative {attempt + 1}/{max_retries + 1}) : {e}")
            if attempt < max_retries:
                time.sleep(1)
            continue
    logger.error("[LLM] Échec après plusieurs tentatives, aucune requête de suivi générée.")
    return []

# Fonction pour générer de nouvelles requêtes d'approfondissement
def generate_new_queries(current_knowledge, initial_query, start_time=None, time_limit=None, num_queries=3, max_retries=3):
    if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
        return []
    logger.info("[LLM] Évaluation des learnings pour générer des requêtes d'approfondissement")
    all_learnings = [fact for learnings in current_knowledge.values() for fact in learnings]
    prompt = (
        f"Tu es un expert en recherche. La date actuelle est {CURRENT_DATE}. Analyse les informations suivantes relatives à '{initial_query}' et identifie "
        f"jusqu'à 3 faits ou données qui méritent d'être approfondis (ex. faits surprenants, spécifiques ou peu explorés), en tenant compte de leur pertinence à {CURRENT_DATE}. "
        f"Pour chaque fait, génère une question spécifique et approfondie pour creuser cet aspect. "
        f"Retourne chaque question sous forme de liste à puces (exemple : '- Question : ...').\n\n"
        f"Informations disponibles :\n" + "\n".join([f"- {learning}" for learning in all_learnings])
    )
    payload = {"prompt": prompt, "max_tokens": 1500, "temperature": 0.7}
    headers = {"Content-Type": "application/json"}
    for attempt in range(max_retries + 1):
        if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
            return []
        try:
            response = session.post(OPENAI_ENDPOINT, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            text = response.json().get("choices", [{}])[0].get("text", "").strip()
            queries = re.findall(r"- Question : (.+)", text)
            if queries:
                logger.info(f"[LLM] Nouvelles requêtes d'approfondissement générées : {queries}")
                return queries[:num_queries]
            else:
                logger.warning(f"[LLM] Aucune requête générée (tentative {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    time.sleep(1)
                continue
        except requests.RequestException as e:
            logger.warning(f"[LLM] Erreur lors de la génération (tentative {attempt + 1}/{max_retries + 1}) : {e}")
            if attempt < max_retries:
                time.sleep(1)
            continue
    logger.error("[LLM] Échec après plusieurs tentatives, aucune nouvelle requête générée.")
    return []

# Fonction principale avec recherche récursive et validation
def deep_research(initial_query, breadth=3, depth=3, time_limit=180):
    global shutdown_flag
    start_time = time.time()
    visited_urls = set()
    current_knowledge = {}
    all_learnings = []
    queries_to_explore = []

    # Formulation de la première requête
    logger.info("Formulation de la première requête par le LLM à partir de l'input donné...")
    prompt_initial = (
        f"Tu es un expert en recherche. La date actuelle est {CURRENT_DATE}. À partir du sujet suivant, formule une question de recherche précise et pertinente, "
        f"adaptée à des moteurs de recherche (ex. Google, Bing), donc courte et directe avec des mots-clés :\n\n"
        f"'{initial_query}'.\nRetourne uniquement la question sous forme de texte sans explication."
    )
    payload_initial = {"prompt": prompt_initial, "max_tokens": 1500, "temperature": 0.7}
    headers = {"Content-Type": "application/json"}
    try:
        response = session.post(OPENAI_ENDPOINT, json=payload_initial, headers=headers, timeout=120)
        response.raise_for_status()
        query = response.json().get("choices", [{}])[0].get("text", "").strip()
        if not query:
            logger.warning("Le LLM n'a pas retourné de requête valide, utilisation de l'input initial.")
            query = initial_query
        logger.info(f"Requête formulée par le LLM : '{query}'")
        queries_to_explore.append(query)
    except Exception as e:
        logger.error(f"Erreur lors de la formulation de la première requête : {e}")
        queries_to_explore.append(initial_query)

    try:
        for iteration in range(depth):
            if shutdown_flag or (time.time() - start_time > time_limit):
                logger.info("Temps limite atteint, passage à la génération du rapport.")
                break
            if not queries_to_explore:
                logger.info("Aucune requête restante à explorer, passage à la génération du rapport.")
                break

            logger.info(f"\n--- Itération {iteration + 1} (Profondeur restante : {depth - iteration}) ---")
            query = queries_to_explore.pop(0)
            logger.info(f"Requête actuelle : '{query}'")
            urls = get_search_results(query, max_results=breadth)
            if not urls:
                logger.info("[SEARXNG] Aucune URL récupérée.")
                continue

            with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
                extract_with_time = partial(extract_content, start_time=start_time, time_limit=time_limit)
                futures = [executor.submit(extract_with_time, url) for url in urls]
                contents = [future.result() for future in as_completed(futures)]

            new_urls = [url for url, content in zip(urls, contents) if content and url not in visited_urls]
            visited_urls.update(new_urls)
            for url, content in zip(urls, contents):
                if shutdown_flag or (time.time() - start_time > time_limit):
                    break
                if content:
                    learnings = analyze_content(content, query, initial_query, start_time=start_time, time_limit=time_limit)
                    if url in current_knowledge:
                        current_knowledge[url].extend(learnings)
                        current_knowledge[url] = list(dict.fromkeys(current_knowledge[url]))
                    else:
                        current_knowledge[url] = learnings
                    all_learnings.extend(learnings)

            # Validation des faits et génération de requêtes de suivi
            follow_up_queries = validate_facts(current_knowledge, initial_query, start_time=start_time, time_limit=time_limit)
            queries_to_explore.extend(follow_up_queries)
            logger.info(f"Requêtes de suivi ajoutées : {follow_up_queries}")

            # Génération de nouvelles requêtes d'approfondissement
            new_queries = generate_new_queries(current_knowledge, initial_query, start_time=start_time, time_limit=time_limit)
            queries_to_explore.extend(new_queries)
            logger.info(f"Requêtes d'approfondissement ajoutées : {new_queries}")

    except KeyboardInterrupt:
        logger.info("Interruption manuelle détectée, arrêt du script.")
        raise

    report = generate_report(initial_query, all_learnings)
    logger.info("\n### Connaissances Accumulées ###")
    if current_knowledge:
        logger.info("Données structurées (learnings/url) :")
        for url, learnings in current_knowledge.items():
            logger.info(f"- URL: {url}\n  Learnings: {learnings}")
    else:
        logger.info("Aucune connaissance accumulée.")
    logger.info("\n### Rapport Final ###")
    logger.info(report)
    return current_knowledge, report

# Fonction pour générer un rapport final
def generate_report(initial_query, all_learnings):
    logger.info("[LLM] Génération du rapport final")
    prompt = (
        f"Tu es un expert en rédaction. La date actuelle est {CURRENT_DATE}. À partir des informations suivantes relatives au sujet '{initial_query}', "
        f"rédige un rapport synthétique et structuré qui répond à la question initiale, en tenant compte de la pertinence des données à {CURRENT_DATE}. "
        f"Ajoute une section finale qui synthétise et donne une perspective globale. Si des données sont incertaines ou nécessitent une vérification, mentionne-le explicitement.\n\n"
        f"Informations disponibles :\n" + "\n".join([f"- {learning}" for learning in all_learnings]) + "\n\n"
        f"Format : Un paragraphe introductif suivi de points clés numérotés."
    )
    payload = {"prompt": prompt, "max_tokens": 25000, "temperature": 0.7}
    headers = {"Content-Type": "application/json"}
    try:
        response = session.post(OPENAI_ENDPOINT, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        report = response.json().get("choices", [{}])[0].get("text", "").strip()
        return report
    except requests.RequestException as e:
        logger.error(f"[LLM] Erreur lors de la génération du rapport : {e}")
        return "Échec de la génération du rapport."

# Point d'entrée
if __name__ == "__main__":
    try:
        initial_query = input("Entrez le sujet de recherche initial : ")
        structured_data, report = deep_research(initial_query, breadth=2, depth=5, time_limit=300)
        print("\n### Données Structurées ###")
        print(json.dumps(structured_data, indent=4, ensure_ascii=False))
        print("\n### Rapport Final ###")
        print(report)
    except KeyboardInterrupt:
        logger.info("Script arrêté par l'utilisateur avec Ctrl+C.")
        sys.exit(0)