"""
Model providers for the benchmark.
Supports Featherless.ai, Together.ai, OpenAI, Ollama, and local models.
"""

import os
import time
import json
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ModelResponse:
    """Response from a model provider."""
    text: str
    model: str
    provider: str
    latency_ms: float
    tokens_used: int = 0
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model identifier."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass
    
    def batch_generate(self, prompts: List[tuple], max_workers: int = 4, **kwargs) -> List[ModelResponse]:
        """Generate responses for multiple prompts in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.generate, sys, usr, **kwargs) for sys, usr in prompts]
            return [f.result() for f in futures]


class OpenAICompatibleProvider(ModelProvider):
    """Provider for OpenAI-compatible APIs (Featherless, Together, OpenAI, etc.)."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ModelResponse:
        client = self._get_client()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            **self.kwargs,
        }
        
        start = time.time()
        try:
            response = client.chat.completions.create(**params, timeout=self.timeout)
            latency_ms = (time.time() - start) * 1000
            
            message = response.choices[0].message
            # Handle reasoning models that output to reasoning_content
            text = message.content or ""
            if not text and hasattr(message, 'reasoning_content') and message.reasoning_content:
                text = message.reasoning_content
            tokens = response.usage.total_tokens if response.usage else 0
            
            return ModelResponse(
                text=text,
                model=self.model,
                provider=self.get_provider_name(),
                latency_ms=latency_ms,
                tokens_used=tokens,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return ModelResponse(
                text="",
                model=self.model,
                provider=self.get_provider_name(),
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def get_model_name(self) -> str:
        return self.model
    
    def get_provider_name(self) -> str:
        if self.base_url:
            if "featherless" in self.base_url:
                return "featherless"
            elif "together" in self.base_url:
                return "together"
            elif "openai" in self.base_url:
                return "openai"
            else:
                return f"openai_compat_{self.base_url.split('//')[1].split('/')[0]}"
        return "openai"


class FeatherlessProvider(OpenAICompatibleProvider):
    """Featherless.ai provider."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("FEATHERLESS_API_KEY")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://api.featherless.ai/v1",
            **kwargs
        )
    
    def get_provider_name(self) -> str:
        return "featherless"


class TogetherProvider(OpenAICompatibleProvider):
    """Together.ai provider."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            **kwargs
        )
    
    def get_provider_name(self) -> str:
        return "together"


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        super().__init__(
            model=model,
            api_key=api_key,
            **kwargs
        )
    
    def get_provider_name(self) -> str:
        return "openai"


class OllamaProvider(ModelProvider):
    """Ollama local provider."""
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 300,
        **kwargs
    ):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ModelResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                **self.kwargs,
            },
        }
        
        start = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.time() - start) * 1000
            
            text = data.get("message", {}).get("content", "")
            tokens = data.get("eval_count", 0)
            
            return ModelResponse(
                text=text,
                model=self.model,
                provider=self.get_provider_name(),
                latency_ms=latency_ms,
                tokens_used=tokens,
                raw_response=data
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return ModelResponse(
                text="",
                model=self.model,
                provider=self.get_provider_name(),
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def get_model_name(self) -> str:
        return self.model
    
    def get_provider_name(self) -> str:
        return "ollama"


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM OpenAI-compatible server."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        **kwargs
    ):
        # Default to no reasoning for all requests
        defaults = {"extra_body": {"chat_template_kwargs": {"enable_reasoning": False}}}
        if "extra_body" in kwargs:
            defaults["extra_body"].update(kwargs.pop("extra_body"))
        kwargs = {**defaults, **kwargs}
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
    
    def get_provider_name(self) -> str:
        return "vllm"


class HuggingFaceTransformersProvider(ModelProvider):
    """Local HuggingFace Transformers model."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        **kwargs
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.kwargs = kwargs
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            if self.dtype == "auto":
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            elif self.dtype == "bf16":
                torch_dtype = torch.bfloat16
            elif self.dtype == "fp16":
                torch_dtype = torch.float16
            elif self.dtype == "fp32":
                torch_dtype = torch.float32
            elif self.dtype == "int8":
                torch_dtype = torch.int8
            else:
                torch_dtype = torch.bfloat16
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch_dtype,
                **self.kwargs
            )
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ModelResponse:
        import torch
        
        self._load_model()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        start = time.time()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=kwargs.get("temperature", self.temperature) > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.time() - start) * 1000
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        tokens_used = outputs.shape[1]
        
        return ModelResponse(
            text=text,
            model=self.model_path.split("/")[-1],
            provider=self.get_provider_name(),
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )
    
    def get_model_name(self) -> str:
        return self.model_path.split("/")[-1]
    
    def get_provider_name(self) -> str:
        return "hf_transformers"


class MockProvider(ModelProvider):
    """Mock provider for testing without API keys."""
    
    def __init__(
        self,
        model: str = "mock-model",
        latency_ms: float = 100.0,
        response_template: Optional[str] = None,
        **kwargs
    ):
        self.model = model
        self.latency_ms = latency_ms
        self.response_template = response_template or self._default_template()
        self.call_count = 0
    
    def _default_template(self) -> str:
        return """[Phonetic Sandbox]
Literal Translation: {literal}

| Chinese | Pinyin | English |
| :--- | :--- | :--- |
| {chinese} | {pinyin} | {english} |

**Tutor's Note:**
- **Grammar/Vocab:** {grammar}
- **Natural Alternative:** {natural}"""
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ModelResponse:
        import time
        
        self.call_count += 1
        time.sleep(self.latency_ms / 1000.0)
        
        # Simple heuristic response based on prompt
        prompt_lower = user_prompt.lower()
        
        if "ni hao ma" in prompt_lower:
            literal = "你好嗎"
            chinese = "你好嗎"
            pinyin = "nǐ hǎo ma"
            english = "you good question_particle"
            grammar = "nǐ (you), hǎo (good), ma (question particle)"
            natural = "你好吗？(nǐ hǎo ma?) - \"How are you?\""
        elif "qu xi shou ba" in prompt_lower:
            literal = "去洗手吧"
            chinese = "去洗手吧"
            pinyin = "qù xǐ shǒu ba"
            english = "go wash hand particle"
            grammar = "qù (go), xǐ (wash), shǒu (hand), ba (sentence-final particle for suggestion)"
            natural = "快去洗手吧 (kuài qù xǐ shǒu ba) - \"Hurry up and go wash your hands\""
        elif "wo yao chi fan le" in prompt_lower:
            literal = "我要吃飯了"
            chinese = "我要吃飯了"
            pinyin = "wǒ yào chī fàn le"
            english = "I want eat rice completed_action"
            grammar = "wǒ (I), yào (want), chī (eat), fàn (rice/meal), le (completed action particle)"
            natural = "我要吃饭了 (wǒ yào chī fàn le) - \"I want to eat now\" / \"I'm going to eat\""
        elif "zen me yang" in prompt_lower:
            literal = "怎麼樣"
            chinese = "怎麼樣"
            pinyin = "zěn me yàng"
            english = "how what kind"
            grammar = "zěn me (how), yàng (kind/type/sort)"
            natural = "怎么样？(zěn me yàng?) - \"How is it?\" / \"How are things?\""
        elif "bu yao qu" in prompt_lower:
            literal = "不要去"
            chinese = "不要去"
            pinyin = "bú yào qù"
            english = "not want go"
            grammar = "bú (not - tone sandhi from bù before yào), yào (want), qù (go)"
            natural = "别去 (bié qù) - \"Don't go\""
        else:
            # Generic fallback
            literal = "示例"
            chinese = "示例"
            pinyin = "shì lì"
            english = "example"
            grammar = "example grammar"
            natural = "示例 (shì lì) - \"Example\""
        
        response = self.response_template.format(
            literal=literal,
            chinese=chinese,
            pinyin=pinyin,
            english=english,
            grammar=grammar,
            natural=natural,
        )
        
        return ModelResponse(
            text=response,
            model=self.model,
            provider=self.get_provider_name(),
            latency_ms=self.latency_ms,
            tokens_used=len(response.split()),
        )
    
    def get_model_name(self) -> str:
        return self.model
    
    def get_provider_name(self) -> str:
        return "mock"


class LlamaCppProvider(ModelProvider):
    """llama.cpp server (llama-server) provider."""
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 300,
        **kwargs
    ):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> ModelResponse:
        # llama.cpp uses completion endpoint
        prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        
        payload = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stop": ["User:", "System:"],
            **self.kwargs,
        }
        
        start = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/completion",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.time() - start) * 1000
            
            text = data.get("content", "")
            tokens = data.get("tokens_predicted", 0)
            
            return ModelResponse(
                text=text,
                model=self.model,
                provider=self.get_provider_name(),
                latency_ms=latency_ms,
                tokens_used=tokens,
                raw_response=data
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return ModelResponse(
                text="",
                model=self.model,
                provider=self.get_provider_name(),
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def get_model_name(self) -> str:
        return self.model
    
    def get_provider_name(self) -> str:
        return "llamacpp"


def create_provider(config: Dict[str, Any]) -> ModelProvider:
    """Factory function to create provider from config dict."""
    provider_type = config.get("type", "featherless").lower()
    
    # Extract common params
    model = config.get("model", config.get("model_path"))
    if not model:
        raise ValueError("Model name required in provider config")
    
    # Remove type and model from config to avoid duplicate kwargs
    kwargs = {k: v for k, v in config.items() if k not in ("type", "model", "model_path")}
    
    # Normalize common aliases
    if "api_base" in kwargs and "base_url" not in kwargs:
        kwargs["base_url"] = kwargs.pop("api_base")
    
    if provider_type == "featherless":
        return FeatherlessProvider(model=model, **kwargs)
    elif provider_type == "together":
        return TogetherProvider(model=model, **kwargs)
    elif provider_type == "openai":
        return OpenAIProvider(model=model, **kwargs)
    elif provider_type == "ollama":
        return OllamaProvider(model=model, **kwargs)
    elif provider_type == "vllm":
        return VLLMProvider(model=model, **kwargs)
    elif provider_type == "llamacpp":
        return LlamaCppProvider(model=model, **kwargs)
    elif provider_type == "hf" or provider_type == "huggingface":
        return HuggingFaceTransformersProvider(model_path=model, **kwargs)
    elif provider_type == "mock":
        return MockProvider(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# Convenience functions for quick setup
def create_featherless(model: str, **kwargs) -> FeatherlessProvider:
    return FeatherlessProvider(model=model, **kwargs)


def create_together(model: str, **kwargs) -> TogetherProvider:
    return TogetherProvider(model=model, **kwargs)


def create_ollama(model: str, **kwargs) -> OllamaProvider:
    return OllamaProvider(model=model, **kwargs)


def create_mock(model: str = "mock-model", **kwargs) -> MockProvider:
    return MockProvider(model=model, **kwargs)