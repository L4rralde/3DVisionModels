import os
from typing import List
import pickle

import requests


class Client:
    def __init__(self, url: str="http://127.0.0.1:8000") -> None:
        self.url = url

    def run(self, img_path_list: List[os.PathLike]) -> dict:
        payload = pickle.loads(img_path_list)

        response = requests.post(
            os.path.join(self.url, "predict"),
            data = payload,
            headers = {"Content-Type": "application/octet-stream"}
        )

        predictions = pickle.loads(response.content)
        return predictions
