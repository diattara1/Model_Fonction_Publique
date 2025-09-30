import asyncio
from transformers import AutoTokenizer
from typing import AsyncGenerator

class StreamingCallbackHandler:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.tokens_buffer = []
        self.skip_tokens = 0
        self.full_text = ""
        
    def set_skip_tokens(self, skip_tokens: int):
        self.skip_tokens = skip_tokens
        
    def put(self, token_tensor):
        new_tokens = token_tensor.tolist()
        if isinstance(new_tokens, list) and len(new_tokens) > 0 and isinstance(new_tokens[0], list):
            new_tokens = new_tokens[0]
            
        for token_id in new_tokens:
            if len(self.tokens_buffer) < self.skip_tokens:
                self.tokens_buffer.append(token_id)
                continue
            decoded_token = self.tokenizer.decode([token_id], skip_special_tokens=True)
            self.full_text += decoded_token
            
    def end(self):
        pass
        
    def get_full_text(self):
        return self.full_text

class AsyncStreamer:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.queue = asyncio.Queue()
        self.skip_tokens = 0
        self.full_text = ""
        
    def set_skip_tokens(self, skip_tokens: int):
        self.skip_tokens = skip_tokens
        
    def put(self, token_tensor):
        new_tokens = token_tensor.tolist()
        if isinstance(new_tokens, list) and len(new_tokens) > 0 and isinstance(new_tokens[0], list):
            new_tokens = new_tokens[0]
            
        for token_id in new_tokens:
            if len(self.queue._queue) < self.skip_tokens:
                continue
            decoded_token = self.tokenizer.decode([token_id], skip_special_tokens=True)
            self.full_text += decoded_token
            asyncio.create_task(self.queue.put(decoded_token))
            
    def end(self):
        asyncio.create_task(self.queue.put(None))
        
    async def __aiter__(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token