#utils/streamer/py
import asyncio
from transformers import AutoTokenizer
from typing import AsyncGenerator

class AsyncStreamer:
    def __init__(self, tokenizer: AutoTokenizer, loop=None):
        self.tokenizer = tokenizer
        self.queue = asyncio.Queue()
        self.skip_tokens = 0
        self.full_text = ""
        # récupère la loop principale (passée en param ou trouvée automatiquement)
        self.loop = loop or asyncio.get_event_loop()
        
    def set_skip_tokens(self, skip_tokens: int):
        self.skip_tokens = skip_tokens
        
    def put(self, token_tensor):
        new_tokens = token_tensor.tolist()
        if isinstance(new_tokens, list) and len(new_tokens) > 0 and isinstance(new_tokens[0], list):
            new_tokens = new_tokens[0]
            
        for token_id in new_tokens:
            decoded_token = self.tokenizer.decode([token_id], skip_special_tokens=True)
            self.full_text += decoded_token
            # thread-safe vers la loop principale
            self.loop.call_soon_threadsafe(self.queue.put_nowait, decoded_token)
            
    def end(self):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, None)
        
    async def __aiter__(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token
