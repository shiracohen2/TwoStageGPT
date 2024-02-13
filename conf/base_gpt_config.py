from dataclasses import dataclass, field, MISSING


@dataclass
class BaseGptConfig:
    api_key: str = MISSING
    azure_endpoint: str = MISSING
    vision_model_deployment_name: str = MISSING
    api_version: str = field(default="2023-05-15")
    max_tokens: int = field(default=600)
    max_rate_limit_retries: int = field(default=5)
    temperature: float = field(default=0.0)
