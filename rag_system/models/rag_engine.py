import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, AsyncGenerator
from config.settings import settings
from models.document_processor import DocumentProcessor
from utils.streamer import AsyncStreamer

class RAGEngine:
    def __init__(self, document_processor: DocumentProcessor):
        self.doc_processor = document_processor
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
    
    def _reformulate_prompt(self, question: str, docs: List[Dict]) -> str:
        """Generate reformulation prompt"""
        context = "\n".join(f"- {d['text'][:500]}..." for d in docs[:3])
        prompt = (
            "Tu es un agent de la fonction publique du Sénégal. "
            "La question suivante n’a pas obtenu de réponse suffisante. "
            "Voici les extraits les plus pertinents trouvés :\n"
            f"{context}\n\n"
            "Reformule la question pour cibler plus précisément l’information manquante, "
            "en t’inspirant du vocabulaire des extraits si pertinent.\n"
            f"Question originale : {question}\n"
            "Nouvelle question :"
        )
        return prompt
    
    def _judge_prompt(self, question: str, docs: List[Dict]) -> str:
        """Generate judgment prompt"""
        if not docs:
            return "REFORMULER"
        
        context = "\n\n".join(f"- {d['text']}" for d in docs)
        prompt = (
            f"Sur une échelle de 1-10, à quel point ces extraits permettent-ils de répondre à la question? "
            f"Réponds avec juste le chiffre.\n\n"
            f"Question: {question}\n\n"
            f"Extraits:\n{context}\n\n"
            f"Score (1-10):"
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
        docs = self.doc_processor.hybrid_search(question, top_k=3)
        
        # Judge if documents are sufficient
        judgment_prompt = self._judge_prompt(question, docs)
        judgment_input = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": judgment_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        judgment_inputs = self.tokenizer(judgment_input, return_tensors="pt").to("cuda")
        with torch.no_grad():
            judgment_out = self.model.generate(
                **judgment_inputs,
                max_new_tokens=3,
                temperature=0.0,
                do_sample=False
            )
        
        input_length = judgment_inputs.input_ids.shape[1]
        judgment_response = self.tokenizer.decode(
            judgment_out[0][input_length:], 
            skip_special_tokens=True
        ).strip()
        
        try:
            score = int(judgment_response)
            verdict = "PERTINENT" if score >= 6 else "REFORMULER"
        except:
            verdict = "REFORMULER"
        
        if verdict == "PERTINENT":
            # Generate answer
            prompt = self._generate_prompt(question, docs)
        else:
            # Reformulate and generate
            reformulate_prompt = self._reformulate_prompt(question, docs)
            reformulate_input = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": reformulate_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            reformulate_inputs = self.tokenizer(reformulate_input, return_tensors="pt").to("cuda")
            with torch.no_grad():
                reformulate_out = self.model.generate(
                    **reformulate_inputs,
                    max_new_tokens=128,
                    temperature=0.0,
                    do_sample=False
                )
            
            input_length = reformulate_inputs.input_ids.shape[1]
            new_question = self.tokenizer.decode(
                reformulate_out[0][input_length:], 
                skip_special_tokens=True
            ).strip()
            
            # Search with new question
            new_docs = self.doc_processor.hybrid_search(new_question, top_k=3)
            prompt = self._generate_prompt(new_question, new_docs)
        
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