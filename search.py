import requests
import os
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

DEBUG = False
log_level = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Configuration du logger pour écrire dans un fichier
logging.basicConfig(
    filename="search.log",  # Nom du fichier de log
    level=logging.INFO,     # Niveau de log (INFO capture tous les messages INFO et supérieurs)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Format des logs
)
logger = logging.getLogger(__name__)

# Configuration des URLs des APIs
SEARXNG_API_URL = "http://searxng:8080/search"  # Nom du service SEARXNG
CRAWL4AI_API_URL = "http://crawl4ai:11235/crawl"  # Nom du service CRAWL4AI
CRAWL4AI_TASK_URL = "http://crawl4ai:11235/task"  # Nom du service CRAWL4AI pour les tâches

# Configuration pour Azure OpenAI
API_KEY = "F1DWSBrZZ7yQLwSccaXlSqdhFN3cQMs0S9yP7HsprLByPY104sXeJQQJ99BCACHYHv6XJ3w3AAAAACOGb6YV"
OPENAI_ENDPOINT = "https://htngu-m85uc1r5-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"

# Token d'authentification pour Crawl4AI
CRAWL4AI_API_TOKEN = "toto"

# Limites de concurrence
CONCURRENCY_LIMIT = 3

# Date actuelle
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")

# Variable pour signaler l'arrêt
shutdown_flag = False

# Cache pour les analyses
analysis_cache = {}

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

# Fonctions utilitaires
def clean_markdown(md):
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"^\s*-+\s*$", "", md, flags=re.MULTILINE)
    md = re.sub(r"https?://\S+", "", md)
    md = re.sub(r"\[.*?\]\(.*?\)", "", md)
    return md.strip()

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

def get_search_results(query, max_results=3):
    if shutdown_flag:
        return []
    logger.info(f"[SEARXNG] Envoi de la requête : '{query}'")
    params = {"q": query, "format": "json", "time_range": "year"}
    try:
        response = session.get(SEARXNG_API_URL, params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
        urls = [result["url"] for result in results[:max_results]]
        logger.info(f"[SEARXNG] URLs récupérées : {len(urls)}")
        return urls
    except requests.RequestException as e:
        logger.error(f"[SEARXNG] Erreur lors de la requête : {e}")
        return []

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

# Analyse avec cache
def analyze_chunk(chunk, query, initial_query, start_time=None, time_limit=None, max_retries=3):
    if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
        return ["Interruption détectée, aucun fait extrait."]
    
    chunk_key = hash(chunk)
    if chunk_key in analysis_cache:
        logger.info(f"[LLM] Cache utilisé pour chunk de longueur {len(chunk)}")
        return analysis_cache[chunk_key]
    
    logger.info(f"[LLM] Analyse d'un chunk de longueur {len(chunk)} pour '{query}'")
    
    # Prompt système assoupli pour garantir des résultats
    system_prompt = (
        f"Tu es un expert en recherche. La date actuelle est {CURRENT_DATE}. Analyse le texte suivant et extrais jusqu'à 3 faits ou données clés "
        f"liés à '{initial_query}'. Si le texte manque de détails spécifiques, formule au moins un fait général basé sur le contexte ou la requête. "
        f"Chaque fait doit être clair, pertinent, et sourcé (mentionne une URL si disponible, sinon indique 'source implicite'). "
        f"Retourne les faits sous forme de liste avec un tiret (-)."
    )
    user_prompt = f"Texte :\n{chunk}"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.5
    }
    logger.debug(f"[LLM] Payload envoyée : {json.dumps(payload, indent=2)}")
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    
    for attempt in range(max_retries + 1):
        try:
            response = session.post(OPENAI_ENDPOINT, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            logger.debug(f"[LLM] Réponse reçue : {text}")
            facts = re.findall(r"- (.+)", text)
            if facts:
                analysis_cache[chunk_key] = facts
                logger.info(f"[LLM] Faits extraits : {facts}")
                return facts
            else:
                # Secours : si aucun fait n'est extrait, retourner un fait générique
                fallback_fact = f"Contenu lié à '{initial_query}' analysé le {CURRENT_DATE}, mais aucun détail spécifique n'a été trouvé (source implicite)."
                analysis_cache[chunk_key] = [fallback_fact]
                logger.info(f"[LLM] Fait de secours utilisé : {fallback_fact}")
                return [fallback_fact]
        except requests.RequestException as e:
            logger.warning(f"[LLM] Erreur d'analyse (tentative {attempt + 1}/{max_retries + 1}) : {e}")
            if attempt < max_retries:
                time.sleep(1)
    
    # Dernier secours en cas d'échec total
    fallback_fact = f"Échec de l'extraction pour '{initial_query}' le {CURRENT_DATE} en raison d'une erreur technique (source implicite)."
    logger.error(f"[LLM] Échec total, retour du fait de secours : {fallback_fact}")
    analysis_cache[chunk_key] = [fallback_fact]
    return [fallback_fact]

def analyze_content(content, query, initial_query, start_time=None, time_limit=None):
    if not content:
        return []
    chunks = chunk_text(content)
    logger.info(f"[LLM] Contenu découpé en {len(chunks)} chunk(s)")
    with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
        analyze_with_time = partial(analyze_chunk, query=query, initial_query=initial_query, start_time=start_time, time_limit=time_limit)
        learnings_chunks = list(executor.map(analyze_with_time, chunks))
    return list(dict.fromkeys([fact for sublist in learnings_chunks for fact in sublist]))

# Validation et génération limitées
def validate_facts(current_knowledge, initial_query, start_time=None, time_limit=None, max_retries=3):
    if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
        return []
    all_learnings = [fact for data in current_knowledge.values() for fact in data["learnings"]]
    system_prompt = (
        f"Tu es un expert en recherche. La date actuelle est {CURRENT_DATE}. Analyse les faits suivants liés à '{initial_query}' et identifie "
        f"jusqu'à 2 faits qui nécessitent une vérification ou un approfondissement. Pour chaque fait, propose une question concise et spécifique. "
        f"Retourne sous forme : '- Fait : ... - Question : ...'."
    )
    user_prompt = f"Faits :\n" + "\n".join([f"- {fact}" for fact in all_learnings])
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.5
    }
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    try:
        response = session.post(OPENAI_ENDPOINT, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        items = re.findall(r"- Fait : (.+?)\s*- Question : (.+)", text)
        return [question for _, question in items[:2]]
    except requests.RequestException as e:
        logger.error(f"[LLM] Erreur lors de la validation : {e}")
        return []

def generate_new_queries(current_knowledge, initial_query, start_time=None, time_limit=None, num_queries=2, max_retries=3):
    if shutdown_flag or (start_time and time_limit and (time.time() - start_time > time_limit)):
        return []
    all_learnings = [fact for data in current_knowledge.values() for fact in data["learnings"]]
    system_prompt = (
        f"Tu es un expert en recherche. La date actuelle est {CURRENT_DATE}. Analyse les faits suivants liés à '{initial_query}' et génère "
        f"jusqu'à 2 questions spécifiques pour approfondir des aspects pertinents à {CURRENT_DATE}. Veille à ce que la question soit bien toujours en lien avec la recherche d'origine '{initial_query}'." 
        f"Retourne sous forme : '- Question : ...'."
    )
    user_prompt = f"Faits :\n" + "\n".join([f"- {fact}" for fact in all_learnings])
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.7
    }
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    try:
        response = session.post(OPENAI_ENDPOINT, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        queries = re.findall(r"- Question : (.+)", text)
        return queries[:num_queries]
    except requests.RequestException as e:
        logger.error(f"[LLM] Erreur lors de la génération : {e}")
        return []

# Recherche récursive
def deep_research(initial_query, breadth=3, depth=3, time_limit=180):
    global shutdown_flag
    start_time = time.time()
    visited_urls = set()
    current_knowledge = {}
    queries_to_explore = [initial_query]

    for iteration in range(depth):
        if shutdown_flag or (time.time() - start_time > time_limit):
            break
        if not queries_to_explore:
            break

        iteration_start = time.time()
        query = queries_to_explore.pop(0)
        logger.info(f"\n--- Itération {iteration + 1} : '{query}' ---")
        urls = get_search_results(query, max_results=breadth)

        with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
            extract_with_time = partial(extract_content, start_time=start_time, time_limit=time_limit)
            futures = [executor.submit(extract_with_time, url) for url in urls]
            contents = [future.result() for future in as_completed(futures)]

        for url, content in zip(urls, contents):
            if shutdown_flag or (time.time() - start_time > time_limit) or (time.time() - iteration_start > 60):
                break
            if content and url not in visited_urls:
                visited_urls.add(url)
                learnings = analyze_content(content, query, initial_query, start_time=start_time, time_limit=time_limit)
                current_knowledge[url] = {
                    "query": query,
                    "depth": iteration + 1,
                    "learnings": learnings
                }
                logger.info(f"[LLM] Learnings pour {url} : {learnings}")

        follow_up_queries = validate_facts(current_knowledge, initial_query, start_time, time_limit)
        queries_to_explore.extend(follow_up_queries[:2])
        new_queries = generate_new_queries(current_knowledge, initial_query, start_time, time_limit)
        queries_to_explore.extend(new_queries[:2])

    all_learnings = list(dict.fromkeys([fact for data in current_knowledge.values() for fact in data["learnings"]]))
    logger.info(all_learnings)
    report = generate_report(initial_query, all_learnings)
    
    logger.info("\n### Données Structurées ###")
    logger.info(json.dumps(current_knowledge, indent=4, ensure_ascii=False))
    logger.info("\n### Rapport Final ###")
    logger.info(report)
    return current_knowledge, report

# Génération du rapport
def generate_report(initial_query, all_learnings):
    system_prompt = (
        f"Tu es un expert en rédaction. La date actuelle est {CURRENT_DATE}. Rédige un rapport synthétique et structuré répondant à '{initial_query}', à destination d'un décideur de l'entreprise "
        f"basé sur les informations suivantes, pertinentes à {CURRENT_DATE}. Ajoute une synthèse finale."
    )
    user_prompt = f"Informations :\n" + "\n".join([f"- {learning}" for learning in all_learnings]) + "\n\nFormat : Très brève introduction + points numérotés avec citations des sources numérotées URL utilisées en bas de page avec une référence renvoyant vers la source, la source doit être l'url complète vers l'article, pas juste le nom de domaine."
    logger.info(user_prompt)
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 4096,
        "temperature": 0.7
    }
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    try:
        response = session.post(OPENAI_ENDPOINT, json=payload, headers=headers, timeout=300)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except requests.RequestException as e:
        logger.error(f"[LLM] Erreur lors de la génération du rapport : {e}")
        return "Échec de la génération du rapport."

# Point d'entrée
if __name__ == "__main__":
    try:
        # Vérifier si un argument est fourni
        if len(sys.argv) < 2:
            # Mode interactif : Demander une entrée utilisateur
            logger.info("Aucun argument fourni. Passage en mode interactif...")
            initial_query = input("Entrez votre requête : ").strip()
            if not initial_query:
                logger.error("Erreur : Requête vide. Arrêt du script.")
                sys.exit(1)
        else:
            # Mode avec argument : Utiliser le premier argument
            initial_query = sys.argv[1]
        # Log de la requête reçue
        logger.info(f"Requête reçue : {initial_query}")
        # Simuler une recherche approfondie
        structured_data, report = deep_research(initial_query, breadth=3, depth=1, time_limit=60)
        # Afficher les résultats
        print("\n### Données Structurées ###")
        print(json.dumps(structured_data, indent=4, ensure_ascii=False))
        print("\n### Rapport Final ###")
        print(report)

        # Écrire les données structurées dans un fichier data.md
        with open("data.md", "w", encoding="utf-8") as data_file:
            data_file.write("### Données Structurées ###\n\n")
            data_file.write(json.dumps(structured_data, indent=4, ensure_ascii=False))
            logger.info("Les données structurées ont été écrites dans data.md")

        # Écrire le rapport final dans un fichier rapport.md
        with open("rapport.md", "w", encoding="utf-8") as rapport_file:
            rapport_file.write("### Rapport Final ###\n\n")
            rapport_file.write(report)
            logger.info("Le rapport final a été écrit dans rapport.md")

    except KeyboardInterrupt:
        logger.info("Script arrêté par l'utilisateur avec Ctrl+C.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}")
        sys.exit(1)