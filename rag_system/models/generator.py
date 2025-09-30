import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, AsyncGenerator
from config.settings import settings
from models.vectorizer import Vectorizer
from utils.streamer import AsyncStreamer

class Generator:
    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL_NAME,
            trust_remote_code=True,
            token=settings.HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
    
    def _generate_prompt(self, question: str, docs: List[Dict]) -> str:
        """Generate prompt for the LLM"""
        context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
        prompt = (
            "Tu es un agent de la fonction publique du Sénégal. "
            "Réponds à la question en 5 phrases maximum.\n"
            "Inclue 1-2 citations EXACTES (entre guillemets) tirées des extraits.\n"
            "À la fin, ajoute une ligne 'Sources: ...' listant fichier:page.\n\n"
            f"Question:\n{question}\n\nExtraits:\n{context}\n\nRéponse:"
        )
        return prompt
    
    async def generate_stream(
        self, 
        question: str, 
        max_tokens: int = 512, 
        temperature: float = 0.1
    ) -> AsyncGenerator[str, None]:
        """Generate response with streaming"""
        # Search for relevant documents
        docs = self.vectorizer.hybrid_search(question, top_k=settings.TOP_K)
        
        # Generate answer
        prompt = self._generate_prompt(question, docs)
        
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        
        # Create streamer
        streamer = AsyncStreamer(self.tokenizer)
        skip_tokens = inputs.input_ids.shape[1]
        streamer.set_skip_tokens(skip_tokens)
        
        # Generate with streaming
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0.0,
            "streamer": streamer
        }
        
        # Start generation in background
        def generate():
            with torch.no_grad():
                self.model.generate(**generation_kwargs)
        
        # Run generation in a separate thread
        import threading
        generation_thread = threading.Thread(target=generate)
        generation_thread.start()
        
        # Yield tokens as they arrive
        async for token in streamer:
            yield token
        
        generation_thread.join()
        
        # Add sources
        sources = [f"{d['source'].split('/')[-1]}:p.{d['page']+1}" for d in docs]
        sources_line = f"\n\nSources: {', '.join(set(sources))}"
        yield sources_line