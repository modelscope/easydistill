import os
from dotenv import load_dotenv
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig


load_dotenv()

class Configuration(BaseModel):
    model_name: str = Field(default=None)
    api_base: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.4)
    max_tokens: int = Field(default=8192)
    api_configs: Optional[Dict[str, Dict[str, str]]] = Field(default=None)
    step_models: Optional[Dict[str, Dict[str, Any]]] = Field(default=None)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
    
        raw_values: Dict[str, Any] = {
            name: configurable.get(name, field.default)
            for name, field in cls.__fields__.items()
        }

        # Handle step_models configuration
        if "step_models" in configurable and isinstance(configurable["step_models"], dict):
            raw_values["step_models"] = configurable["step_models"]

        values = {k: v for k, v in raw_values.items() if v is not None}
        
        # Handle API configuration based on model name
        if "api_configs" in values and "model_name" in values:
            model_name = values["model_name"]
            api_configs = values["api_configs"]
            
            # Find the API config for this model, or use default
            model_api_config = api_configs.get(model_name, api_configs.get("default", {}))
            
            # Set api_base and api_key based on the model's API config
            if "api_base" not in values and "api_base" in model_api_config:
                values["api_base"] = os.getenv(model_api_config["api_base"], model_api_config["api_base"])
            
            if "api_key" not in values and "api_key_env" in model_api_config:
                values["api_key"] = os.getenv(model_api_config["api_key_env"], model_api_config["api_key_env"])
        
        return cls(**values)
