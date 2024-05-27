from typing import Literal
import pybuda
from phi3.configuration_phi3 import Phi3Config
from phi3.modeling_phi3 import Phi3ForCausalLM
from pybuda._C.backend_api import BackendDevice
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer

variants = Literal["mini"]
contexts = Literal["4k"]

def run_phi3_pybuda_pipeline(prompt: str, _variant: variants = "mini", _context: contexts = "4k"):
    available_devices = pybuda.detect_available_devices()
    print(f"Available devices: {available_devices}")

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    print(f"Compiler config: {compiler_cfg}")

    model_ckpt = f"microsoft/Phi-3-{_variant}-{_context}-instruct"
    config = Phi3Config.from_pretrained(model_ckpt)
    config_dict = config.to_dict()
    config_dict["_attn_implementation"] = "eager"
    config = Phi3Config(**config_dict)
    print(f"Model config: {config}")

    model = Phi3ForCausalLM.from_pretrained(model_ckpt, config=config)
    print(f"Model: {model}")

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    print(f"Tokenizer: {tokenizer}")

    text_generator = pybuda_pipeline("text-generation", model=model, tokenizer=tokenizer)
    print(f"Pipeline: {text_generator}")

    result = text_generator(prompt)
    print("Result:", result)

if __name__ == "__main__":
    prompt = "Why do birds sing?"
    run_phi3_pybuda_pipeline(prompt)
    
