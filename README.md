# hackaton-search

# Lancement du script:

python search.py

# docker compose à utiliser :

```yaml
version: "3.8"
services:
  searxng:
    image: dockreg.renzan.fr/searxng:latest
    container_name: searxng
    restart: unless-stopped
    ports:
      - 8081:8080
    volumes:
      - settings_searxng:/etc/searxng
    environment:
      - BASE_URL=http://127.0.0.1:8081
      - INSTANCE_NAME=searxng
  crawl4ai:
    image: unclecode/crawl4ai:basic-amd64
    restart: unless-stopped
    ports:
      - 11235:11235
    environment:
      CRAWL4AI_API_TOKEN: toto
volumes:
  settings_searxng: null
networks: {}

