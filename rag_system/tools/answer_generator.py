# tools/answer_generator.py
import os
import asyncio
import threading
from typing import List, Dict, Tuple
from transformers import TextIteratorStreamer

from models.generator import THINK_OPEN, THINK_CLOSE

class AnswerGenerator:
    """
    Génération finale basée sur des extraits (RAG).
    La fin de stream (STATUS_DONE) est gérée par l'AGENT, pas ici.
    """

    def __init__(self, generator):
        self.gen = generator
        self.tok = generator.tokenizer
        self.model = generator.model

    def _scrub(self, text: str) -> str:
        lower = (text or "").lower()
        dangerous = [
            "ignore previous instructions",
            "system prompt",
            "prompt système",
            "reveal your instructions",
        ]
        if any(p in lower for p in dangerous):
            return "[Extrait neutralisé pour sécurité]"
        return text

    def _build_prompt(self, question: str, docs: List[Dict]) -> str:
        context = "\n\n".join(
            f"[{i+1}] {self._scrub(d['text'])}"
            for i, d in enumerate(docs)
        )
        return (
            "Tu es un assistant spécialisé dans la Fonction publique du Sénégal.\n"
            "Réponds en 5 phrases maximum, clairement et sans inventer.\n"
            "- Si tu cites un texte, mets la citation EXACTE entre guillemets.\n"
            "- Utilise uniquement les extraits fournis (pas d'autres sources implicites).\n\n"
            f"Question:\n{question}\n\nExtraits:\n{context}\n\nRéponse:"
        )

    async def stream_answer(
        self,
        question: str,
        docs: List[Dict],
        conv_id: int,
        use_history: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.1
    ):
        rag_prompt = self._build_prompt(question, docs)
        messages = self.gen._messages(rag_prompt, conv_id if use_history else None)
        inputs = self.gen._tokenize_messages(messages)

        streamer = TextIteratorStreamer(
            self.tok,
            skip_prompt=True,
            skip_special_tokens=True
        )

        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            streamer=streamer,
            bad_words_ids=self.gen._bad_words_ids(),
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

        # Ajout propre des sources (UNE seule ligne, dédupliquée)
        sources = [
            f"{os.path.basename(d['source'])}:p.{d['page']+1}"
            for d in docs
            if 'source' in d and 'page' in d
        ]
        if sources:
            uniq = ", ".join(sorted(set(sources)))
            yield f"\n\n📚 Sources: {uniq}"

        if use_history:
            final_response = "".join(collected_visible).strip()
            self.gen._save_message(conv_id, "user", question)
            # on ajoute également la ligne sources si elle existe
            if sources:
                final_response = (final_response + f"\n\n📚 Sources: {uniq}").strip()
            self.gen._save_message(conv_id, "assistant", final_response)

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
                    open_pos = buf.rfind(THINK_OPEN, 0, max(i, 0))
                    keep_from = open_pos if open_pos != -1 else i
                    return ("".join(out), buf[keep_from:], True)
                else:
                    i = end + len(THINK_CLOSE)
                    in_think = False
        return ("".join(out), "", in_think)
