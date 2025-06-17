from __future__ import annotations

import json
from typing import Dict, Iterable, List, Sequence

import requests


class OllamaRemoteClient:
    
    def __init__(self, base_url: str, model_name: str) -> None:
        self.base_url: str = base_url.rstrip("/") or "http://localhost:11434"
        self.model_name: str = model_name

    # --------------------------------------------------------------------- #
    #                           Public methods                               #
    # --------------------------------------------------------------------- #
    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        stream: bool = False,
        **generation_params
    ) -> str | Iterable[str]:
        
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            **generation_params,
        }

        if not stream:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["message"]["content"]

        return self._chat_stream(url, payload)

    def stop(self, *, timeout: int = 15) -> dict:
        
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model_name, "prompt": "", "keep_alive": 0}
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def stop_formatted(self, *, timeout: int = 15) -> str:
        resp = self.stop(timeout=timeout)
        model = resp.get("model", self.model_name)
        done = resp.get("done_reason", "N/D")
        ts = resp.get("created_at", "N/D")
        return f"Modello «{model}» scaricato ({done}) - creato: {ts}"

    def list_models(self) -> List[dict]:
        url = f"{self.base_url}/api/tags"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json().get("models", [])

    # --------------------------------------------------------------------- #
    #                         Private helpers                                #
    # --------------------------------------------------------------------- #
    def _chat_stream(self, url: str, payload: dict) -> Iterable[str]:
        with requests.post(url, json=payload, stream=True) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                    text = data.get("message", {}).get("content", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue