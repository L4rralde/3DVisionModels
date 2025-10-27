from threading import Thread

from inference.mapanything import MapanythingInference
from api import Server


mathing = MapanythingInference()
server = Server(mathing)
Thread(target = server, daemon=True).start()
input("Press entir to exit...\n")
