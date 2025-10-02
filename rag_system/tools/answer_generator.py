# tools/answer_generator.py
import os
import asyncio
import threading
from typing import List, Dict, Tuple
from transformers import TextIteratorStreamer

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

class AnswerGenerator:
    """
    Génération finale basée sur des extraits (RAG).
    - stream_answer(..., use_history=False) : comportement historique
    - stream_answer(..., use_history=True)  : inclut l'historique LLM (multi-tour)
    Filtrage des tokens <think> en streaming.
    """

    def __init__(self, generator):
        self.gen = generator            # models.generator.Generator
        self.tok = generator.tokenizer
        self.model = generator.model

    def _scrub(self, text: str) -> str:
        lower = text.lower()
        if any(k in lower for k in ["ignore previous instructions", "system prompt", "prompt système", "<think>"]):
            return "[Extrait neutralisé pour sécurité]"
        return text

    def _build_prompt(self, question: str, docs: List[Dict]) -> str:
        context = "\n\n".join(f"[{i+1}] {self._scrub(d['text'])}" for i, d in enumerate(docs))
        return (
            "Tu es un assistant spécialisé dans la Fonction publique du Sénégal. "
            "Réponds en 5 phrases maximum, clairement et sans inventer.\n"
            "- Si tu cites un texte, mets la citation EXACTE entre guillemets.\n"
            "- Utilise uniquement les extraits fournis (pas d'autres sources implicites).\n"
            "- À la fin, ajoute une ligne '📚 Sources: fichier:page'.\n\n"
            f"Question:\n{question}\n\nExtraits:\n{context}\n\nRéponse:"
        )

    async def stream_answer(self, question: str, docs: List[Dict], use_history: bool = False,
                            max_new_tokens=512, temperature=0.1):
        """
        Stream de la réponse finale RAG.
        - use_history = True → on injecte self.gen.history avant le user turn.
        - À la fin, on ajoute la réponse + sources dans l'historique si use_history=True.
        """
        user_turn = self._build_prompt(question, docs)

        messages = [{"role": "system", "content": "Tu es un assistant spécialisé dans la Fonction publique du Sénégal."}]
        if use_history and self.gen.history:
            messages.extend(self.gen.history)
        messages.append({"role": "user", "content": user_turn})

        text = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
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

        # Ajouter les sources à la fin
        sources = [f"{os.path.basename(d['source'])}:p.{d['page']+1}" for d in docs if 'source' in d and 'page' in d]
        if sources:
            line = "📚 Sources: " + ", ".join(sorted(set(sources)))
            collected_visible.append(f"\n\n{line}")
            yield f"\n\n{line}"

        # MàJ de la mémoire si demandé
        if use_history:
            final_text = "".join(collected_visible).strip()
            self.gen.add_to_history("user", question)          # on enregistre la *vraie* question utilisateur
            self.gen.add_to_history("assistant", final_text)   # réponse + sources

    # --- filtre thinking en streaming ---
    def _filter_think_from_buffer(self, buf: str, in_think: bool) -> Tuple[str, str, bool]:
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
                    return ("".join(out), buf[i - len(THINK_OPEN):] if i - len(THINK_OPEN) >= 0 else buf, True)
                else:
                    i = end + len(THINK_CLOSE)
                    in_think = False
        return ("".join(out), "", in_think)
