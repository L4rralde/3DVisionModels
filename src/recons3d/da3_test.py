from threading import Thread


from inference.da3 import Da3Inference
from api import Server


da3 = Da3Inference("DA3-GIANT")
server = Server(da3)
Thread(target = server, daemon=True).start()
input("Press entir to exit...\n")
