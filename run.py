import ssl
import streamlit.web.bootstrap
from streamlit.web.server import Server

def run():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain('cert.pem', 'key.pem')
    
    Server.ssl_context = ssl_context
    streamlit.web.bootstrap.run("app.py", "", [], [])

if __name__ == "__main__":
    run()
