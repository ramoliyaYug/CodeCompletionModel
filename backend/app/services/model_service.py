import asyncio
import gc
import json
import os
import threading
import time
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any

from backend.app.config import settings
from backend.app.utils.logger import get_logger, sanitize_code_for_logging

logger = get_logger("model_service")


class ModelStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ModelService:
    """
    Singleton service managing the lifecycle, loading, and inference of the
    fine-tuned StarCoder2-3B QLoRA model.
    """

    _instance: Optional["ModelService"] = None

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._init_state()
        return cls._instance

    def _init_state(self) -> None:
        self.status: ModelStatus = ModelStatus.UNLOADED
        self.status_message: str = "Model is not loaded."
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str = "cpu"
        self.quantization: Optional[str] = None
        self.base_model_name: str = settings.BASE_MODEL_NAME_OR_PATH
        self.adapter_dir: str = settings.ADAPTER_DIR
        self.lock = asyncio.Lock()
        self._load_lock = threading.Lock()

    def _detect_base_model_name(self) -> str:
        """Attempts to read base_model_name_or_path from adapter_config.json if present."""
        adapter_cfg_file = Path(self.adapter_dir) / "adapter_config.json"
        if adapter_cfg_file.exists():
            try:
                with open(adapter_cfg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "base_model_name_or_path" in data and data["base_model_name_or_path"]:
                        return data["base_model_name_or_path"]
            except Exception as e:
                logger.warning(f"Could not read base_model_name_or_path from {adapter_cfg_file}: {e}")
        return settings.BASE_MODEL_NAME_OR_PATH

    async def initialize(self) -> None:
        """Asynchronously triggers model loading if configured for auto-load."""
        if settings.AUTO_LOAD_MODEL and self.status == ModelStatus.UNLOADED:
            asyncio.create_task(self.load_model())

    async def load_model(self) -> bool:
        """Loads tokenizer, base model, and LoRA adapter into memory."""
        async with self.lock:
            if self.status == ModelStatus.READY:
                logger.info("Model is already loaded and ready.")
                return True

            self.status = ModelStatus.LOADING
            self.status_message = "Loading tokenizer and model weights..."
            logger.info("Starting model loading process...")

            # Run heavy loading in a separate thread so the event loop stays responsive
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(None, self._load_model_sync)
            return success

    def _load_model_sync(self) -> bool:
        with self._load_lock:
            try:
                if settings.MOCK_INFERENCE:
                    logger.info("MOCK_INFERENCE is enabled. Operating in simulated inference mode.")
                    self.device = "mock"
                    self.quantization = "none"
                    self.status = ModelStatus.READY
                    self.status_message = "Mock model ready for inference."
                    return True

                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
                from peft import PeftModel

                self.base_model_name = self._detect_base_model_name()
                logger.info(f"Target base model: {self.base_model_name}")
                logger.info(f"Adapter path: {self.adapter_dir}")

                # 1. Load Tokenizer
                tokenizer_path = self.adapter_dir if Path(self.adapter_dir).exists() else self.base_model_name
                logger.info(f"Loading tokenizer from: {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"

                # 2. Determine Device & Quantization
                has_cuda = torch.cuda.is_available() and settings.DEVICE != "cpu"
                if has_cuda:
                    self.device = "cuda:0"
                    use_bf16 = torch.cuda.is_bf16_supported()
                    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

                    if settings.USE_4BIT_QUANTIZATION:
                        logger.info("Configuring 4-bit NF4 BitsAndBytes quantization...")
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=compute_dtype,
                        )
                        self.quantization = "4-bit NF4"
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.base_model_name,
                            quantization_config=bnb_config,
                            torch_dtype=compute_dtype,
                            device_map={"": 0},
                        )
                    else:
                        self.quantization = "16-bit (bf16/fp16)"
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.base_model_name,
                            torch_dtype=compute_dtype,
                            device_map={"": 0},
                        )
                else:
                    self.device = "cpu"
                    self.quantization = "float32"
                    logger.warning("CUDA is not available or disabled; loading base model on CPU...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                    )

                # 3. Attach LoRA Adapter
                if Path(self.adapter_dir).exists():
                    logger.info(f"Attaching LoRA adapter from {self.adapter_dir}...")
                    self.model = PeftModel.from_pretrained(self.model, self.adapter_dir)
                else:
                    logger.warning(
                        f"Adapter directory not found at {self.adapter_dir}. Running base model without adapter."
                    )

                self.model.eval()
                self.model.config.use_cache = True

                self.status = ModelStatus.READY
                self.status_message = "Fine-tuned StarCoder2-3B model is loaded and ready."
                logger.info(f"Model successfully loaded. Device: {self.device}, Quantization: {self.quantization}")
                return True

            except Exception as e:
                logger.exception(f"Failed to load model: {e}")
                self.status = ModelStatus.ERROR
                self.status_message = f"Model load failure: {str(e)}"
                return False

    def unload_model(self) -> None:
        """Unloads model from memory and clears CUDA cache."""
        with self._load_lock:
            logger.info("Unloading model and clearing GPU cache...")
            self.model = None
            self.tokenizer = None
            self.status = ModelStatus.UNLOADED
            self.status_message = "Model unloaded."

            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Model unloaded successfully.")

    def is_ready(self) -> bool:
        return self.status == ModelStatus.READY

    def _truncate_input_prompt(self, code: str, max_new_tokens: int) -> str:
        """Truncates code prompt from the left if it exceeds context limit."""
        if self.tokenizer is None:
            return code

        max_prompt_tokens = max(1, settings.MAX_CONTEXT_LENGTH - max_new_tokens)
        token_ids = self.tokenizer.encode(code, add_special_tokens=False)
        if len(token_ids) > max_prompt_tokens:
            truncated_ids = token_ids[-max_prompt_tokens:]
            code = self.tokenizer.decode(
                truncated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        return code

    def _apply_stop_tokens(self, text: str, stop_tokens: Optional[List[str]]) -> str:
        if not stop_tokens:
            return text
        earliest_idx = len(text)
        found = False
        for st in stop_tokens:
            if not st:
                continue
            idx = text.find(st)
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx
                found = True
        return text[:earliest_idx] if found else text

    def generate_completion(
        self,
        code: str,
        max_new_tokens: int = 32,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_tokens: Optional[List[str]] = None,
        fim_suffix: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Generates a single code completion.
        
        Returns:
            (completion_text: str, latency_ms: int)
        """
        start_time = time.perf_counter()

        if settings.MOCK_INFERENCE:
            # Deterministic simulation for tests
            latency_ms = int((time.perf_counter() - start_time) * 1000) + 12
            mock_comp = " a + b" if "add" in code or "return" in code else "\n    pass"
            return mock_comp, latency_ms

        if not self.is_ready():
            raise RuntimeError(f"Model is not ready. Current status: {self.status.value} ({self.status_message})")

        import torch

        # Handle Fill-In-The-Middle (FIM) if suffix is provided
        prompt_text = code
        if fim_suffix:
            prompt_text = f"<fim_prefix>{code}<fim_suffix>{fim_suffix}<fim_middle>"

        prompt_text = self._truncate_input_prompt(prompt_text, max_new_tokens)

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_CONTEXT_LENGTH,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "repetition_penalty": settings.REPETITION_PENALTY,
        }

        if temperature > 0:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_length:]

        completion = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        completion = self._apply_stop_tokens(completion, stop_tokens)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            f"Completion generated in {latency_ms}ms: '{sanitize_code_for_logging(completion)}'"
        )
        return completion, latency_ms

    def generate_top_k(
        self,
        code: str,
        k: int = 3,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_tokens: Optional[List[str]] = None,
    ) -> Tuple[List[str], int]:
        """
        Generates k diverse completion candidates.
        
        Returns:
            (suggestions: List[str], latency_ms: int)
        """
        start_time = time.perf_counter()

        if settings.MOCK_INFERENCE:
            latency_ms = int((time.perf_counter() - start_time) * 1000) + 25
            suggestions = [
                " a + b",
                " b + a",
                " sum([a, b])",
            ][:k]
            return suggestions, latency_ms

        if not self.is_ready():
            raise RuntimeError(f"Model is not ready. Current status: {self.status.value}")

        import torch

        prompt_text = self._truncate_input_prompt(code, max_new_tokens)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_CONTEXT_LENGTH,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "do_sample": True,
            "temperature": max(0.2, temperature),
            "top_p": top_p,
            "num_return_sequences": k,
            "repetition_penalty": settings.REPETITION_PENALTY,
        }

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        suggestions: List[str] = []
        seen = set()

        for seq in outputs:
            gen_ids = seq[prompt_length:]
            text = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            text = self._apply_stop_tokens(text, stop_tokens)
            if text not in seen:
                seen.add(text)
                suggestions.append(text)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return suggestions, latency_ms

    async def generate_stream(
        self,
        code: str,
        max_new_tokens: int = 32,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_tokens: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously streams generated tokens using Hugging Face TextIteratorStreamer.
        """
        if settings.MOCK_INFERENCE:
            mock_tokens = [" ", "a", " +", " b"]
            for token in mock_tokens:
                await asyncio.sleep(0.05)
                yield token
            return

        if not self.is_ready():
            raise RuntimeError(f"Model is not ready. Status: {self.status.value}")

        from transformers import TextIteratorStreamer

        prompt_text = self._truncate_input_prompt(code, max_new_tokens)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_CONTEXT_LENGTH,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "repetition_penalty": settings.REPETITION_PENALTY,
        }

        if temperature > 0:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        else:
            gen_kwargs["do_sample"] = False

        # Run model.generate in a background thread
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        for token_chunk in streamer:
            accumulated += token_chunk
            if stop_tokens and any(st in accumulated for st in stop_tokens):
                break
            yield token_chunk
            await asyncio.sleep(0)  # Yield control to event loop

        thread.join()

    def get_gpu_memory(self) -> Optional[Dict[str, Any]]:
        """Returns GPU VRAM consumption stats if CUDA is active."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    "allocatedGB": round(allocated, 2),
                    "reservedGB": round(reserved, 2),
                    "totalGB": round(total, 2),
                    "gpuName": torch.cuda.get_device_name(0),
                }
        except Exception:
            pass
        return None

    def get_model_info(self) -> Dict[str, Any]:
        """Provides metadata about the model, adapter, and device configuration."""
        return {
            "model": settings.MODEL_NAME,
            "adapter": Path(self.adapter_dir).name,
            "version": settings.MODEL_VERSION,
            "language": "python",
            "device": self.device,
            "quantization": self.quantization or "none",
            "status": self.status.value,
            "details": {
                "baseModel": self.base_model_name,
                "adapterDir": self.adapter_dir,
                "statusMessage": self.status_message,
                "maxContextLength": settings.MAX_CONTEXT_LENGTH,
            },
        }


model_service = ModelService()
