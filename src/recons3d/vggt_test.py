from threading import Thread

from inference.vggt import VggtInference
from api import Server


vggt = VggtInference()
server = Server(vggt)
Thread(target=server, daemon=True).start()
input("Press Enter to exit...\n")
