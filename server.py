import http.server
import socketserver
import subprocess
from urllib.parse import urlparse, parse_qs

PORT = 8000

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Affiche le chemin reçu dans les logs pour déboguer
        print(f"Reçu : {self.path}")
        
        # Analyse l'URL pour extraire le chemin et les paramètres
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip('/')  # Chemin sans slash final
        query_params = parse_qs(parsed_url.query)  # Paramètres sous forme de dictionnaire
        
        if path == '/search':
            # Extraire les arguments depuis les paramètres
            arg1 = query_params.get('arg1', [None])[0]  # Valeur de 'arg1' ou None
            
            if not arg1:
                self.send_response(400)  # Bad Request
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Erreur : Aucun argument fourni.")
                return
            
            # Débogage : Afficher l'argument extrait
            print(f"Argument extrait : {arg1}")
            
            # Construire la commande avec les arguments
            command = ["python3", "search.py", arg1]
            
            # Exécuter le script avec les arguments
            result = subprocess.run(command, capture_output=True, text=True)
            output = result.stdout if result.returncode == 0 else result.stderr
            
            # Envoyer la réponse HTTP
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(output.encode('utf-8'))
        else:
            # Endpoint non trouvé
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Endpoint non trouvé : {self.path}".encode('utf-8'))

with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()