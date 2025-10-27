import pickle

from fastapi import FastAPI, Request, Response
import uvicorn


class Server:
    def __init__(self, model):
        self.model = model
        self.app = FastAPI()

    def __call__(self):
        @self.app.post("/predict")
        async def predict(request: Request) -> object:
            body = await request.body()
            imgs = pickle.loads(body)
            predictions = self.model.infer(imgs)
            response = Response(
                content=pickle.dumps(predictions),
                media_type = "application/octect-stream"
            )
            return response
        
        uvicorn.run(self.app, host="127.0.0.1", port=8000)
