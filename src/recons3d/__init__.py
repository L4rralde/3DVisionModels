import os
import sys
import importlib
from threading import Thread

from .api import Server, Client


__inference_path = os.path.join(os.path.dirname(__file__), 'inference')
available_models = os.listdir(__inference_path)


def get_model_cls(model_name: str):
    if not model_name in available_models:
        raise ValueError(f"model {model_name} is not supported. Try one of the following instead: {available_models}")
    module_name = f"inference.{model_name}"

    module_path = os.path.join(__inference_path, model_name, '__init__.py')
    if not os.path.exists(module_path):
        raise ValueError(f"Unable to find module for {model_name} model.")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
        #module = importlib.import_module(module_name) Duplicated import
        sys.modules[module_name] = module
    except Exception as e:
        raise RuntimeError(
            f"Module import failed for {model_name}. "
            f"Check your environment. Original error: {e}"
        ) from e
        #The from e preserves the original traceback.
        # For instance,     
        # Here, the exception context is being lost
        #raise RuntimeError(
        #    e,
        #    f"Please check if you are using the right environment for {model_name}"
        #)


    class_name = f"{model_name.capitalize()}Inference"
    try:
        cls = getattr(module, class_name)
    except Exception as e:
        raise RuntimeError(e)
    
    return cls


def make_server(model) -> Thread:
    server = Server(model)
    thread = Thread(target=server, daemon=True)

    return thread
