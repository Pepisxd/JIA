from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str, str], "LocalEduModel"] = {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class LocalEduModel:
    def __init__(self, model_base: str, adapter_dir: str, device_map: str = "auto") -> None:
        self.model_base = model_base
        self.adapter_dir = str(Path(adapter_dir))
        self.device_map = device_map

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Dependencias de modelo local faltantes. Instala transformers, peft, bitsandbytes y torch."
            ) from exc

        self._torch = torch
        local_files_only = _env_bool("LOCAL_MODEL_LOCAL_FILES_ONLY", False)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Tokenizer del adapter para respetar special tokens del fine-tuning.
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_base,
            device_map=self.device_map,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            local_files_only=local_files_only,
        )
        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir, local_files_only=local_files_only)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 700,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        if "[/INST]" in decoded:
            return decoded.split("[/INST]", 1)[-1].strip()
        return decoded.strip()


def get_local_model(
    model_base: str,
    adapter_dir: str,
    device_map: str = "auto",
) -> LocalEduModel:
    key = (model_base, str(Path(adapter_dir)), device_map)
    with _LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = LocalEduModel(model_base=model_base, adapter_dir=adapter_dir, device_map=device_map)
        return _MODEL_CACHE[key]
