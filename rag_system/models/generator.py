# models/generator.py
import re
import asyncio
import threading
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from config.settings import settings

# Persona (tu peux le garder tel quel)
SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé dans la Fonction publique du Sénégal. "
    "Tu aides uniquement sur des questions juridiques, administratives et réglementaires "
    "(lois, décrets, statuts, procédures, carrières, rémunérations, concours, etc.). "
    "Si la question sort de ce cadre, demande poliment de la reformuler dans le périmètre."
)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

def strip_think(text: str) -> str:
    """Retire les blocs <think>...</think> d'un texte complet (mode non-stream)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class Generator:
    """
    Générateur LLM avec deux modes :
    - Non mémoire (backward-compatible) : simple_answer / stream_generate
    - Multi-tour (avec mémoire) : simple_answer_mt / stream_generate_mt

    Dans les deux cas, on active enable_thinking=True et on *filtre* les tokens <think> en streaming.
    Le switch /think /no_think fonctionne : ce sont juste des tags dans le texte user.
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL_NAME,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        # --- mémoire multi-tour ---
        # On stocke la conversation sous forme [{"role":"user"/"assistant"/"system","content": "..."}]
        self.history: List[Dict[str, str]] = []

    # ------------------------
    # Utilitaires mémoire
    # ------------------------
    def reset_history(self):
        self.history = []

    def add_to_history(self, role: str, content: str):
        assert role in ("system", "user", "assistant"), "role invalide"
        self.history.append({"role": role, "content": content})

    def _messages(self, user_prompt: str, use_history: bool) -> List[Dict[str, str]]:
        """
        Construit la liste de messages pour Qwen :
        system + (history optionnelle) + user
        """
        msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if use_history and self.history:
            msgs.extend(self.history)
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    def _tokenize_messages(self, messages: List[Dict[str, str]]):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 👉 thinking actif (les /think /no_think du user pilotent à chaque tour)
        )
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    # ------------------------
    # Non-mémoire (compat)
    # ------------------------
    def simple_answer(self, prompt: str, max_new_tokens=512, temperature=0.1) -> str:
        """
        Réponse directe *sans* mémoire (comportement historique).
        """
        inputs = self._tokenize_messages(self._messages(prompt, use_history=False))
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
            )[0]
        input_len = inputs.input_ids.shape[1]
        resp_ids = out_ids[input_len:]
        raw = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
        return strip_think(raw)

    async def stream_generate(self, prompt: str, max_new_tokens=1024, temperature=0.1):
        """
        Streaming direct *sans* mémoire (comportement historique).
        Filtrage des tokens <think> : on n'affiche rien tant que </think> n'est pas passé.
        """
        inputs = self._tokenize_messages(self._messages(prompt, use_history=False))
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        in_think = False
        buffer = ""
        for chunk in streamer:
            visible, buffer, in_think = self._filter_think_from_buffer(buffer + chunk, in_think)
            if visible:
                yield visible
            await asyncio.sleep(0)

        thread.join()

    # ------------------------
    # Multi-tour (mémoire)
    # ------------------------
    def simple_answer_mt(self, prompt: str, max_new_tokens=512, temperature=0.1) -> str:
        """
        Réponse directe *avec* mémoire : ajoute la question et la réponse à self.history.
        """
        messages = self._messages(prompt, use_history=True)
        inputs = self._tokenize_messages(messages)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
            )[0]
        input_len = inputs.input_ids.shape[1]
        resp_ids = out_ids[input_len:]
        raw = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
        resp = strip_think(raw)

        # MàJ mémoire (on stocke le prompt *brut* de l'utilisateur)
        self.add_to_history("user", prompt)
        self.add_to_history("assistant", resp)
        return resp

    async def stream_generate_mt(self, prompt: str, max_new_tokens=1024, temperature=0.1):
        """
        Streaming *avec* mémoire : à la fin, ajoute la question + la réponse dans self.history.
        """
        messages = self._messages(prompt, use_history=True)
        inputs = self._tokenize_messages(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        in_think = False
        buffer = ""
        collected_visible = []

        for chunk in streamer:
            visible, buffer, in_think = self._filter_think_from_buffer(buffer + chunk, in_think)
            if visible:
                collected_visible.append(visible)
                yield visible
            await asyncio.sleep(0)

        thread.join()

        # MàJ mémoire
        full_resp = "".join(collected_visible).strip()
        self.add_to_history("user", prompt)
        self.add_to_history("assistant", full_resp)

    # ------------------------
    # Filtrage des blocs thinking en streaming
    # ------------------------
    def _filter_think_from_buffer(self, buf: str, in_think: bool) -> Tuple[str, str, bool]:
        """
        Retire les blocs <think>...</think> du buffer (même s'ils arrivent morcelés).
        Retourne (portion_visible, nouveau_buffer, in_think_state).
        """
        out = []
        i = 0
        while i < len(buf):
            if not in_think:
                start = buf.find(THINK_OPEN, i)
                if start == -1:
                    out.append(buf[i:])
                    i = len(buf)
                else:
                    out.append(buf[i:start])
                    i = start + len(THINK_OPEN)
                    in_think = True
            else:
                end = buf.find(THINK_CLOSE, i)
                if end == -1:
                    # On reste dans un bloc <think> incomplet → garder le reste en buffer
                    # Retourner rien de visible pour l’instant
                    return ("".join(out), buf[i - len(THINK_OPEN):] if i - len(THINK_OPEN) >= 0 else buf, True)
                else:
                    # on saute le bloc think
                    i = end + len(THINK_CLOSE)
                    in_think = False

        # tout consommé, rien à garder
        return ("".join(out), "", in_think)
