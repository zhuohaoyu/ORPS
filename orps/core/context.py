import copy
from typing import Dict, Any
import logging

class Context:
    # context class for a single evaluation
    def __init__(self, config) -> None:
        self.config = config
        self.dataset = None
        self.predictions = {}
        self.results = {}
        self.logger = logging.getLogger(__name__)
        self.log_path = None

    def get_safe_config(self) -> Dict:
        def hide_sensitive_info(d: Any) -> Any:
            # Define keys that are considered sensitive
            sensitive_keys = ['key', 'secret', 'proxy', 'openai_api_base', 'openai_key', 'base_url']
            
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, dict):
                        d[k] = hide_sensitive_info(v)
                    elif isinstance(v, list):
                        if any(sensitive_key in k.lower() for sensitive_key in sensitive_keys):
                            d[k] = 'hidden'
                        else:
                            d[k] = [hide_sensitive_info(item) for item in v]
                    elif any(sensitive_key in k.lower() for sensitive_key in sensitive_keys):
                        d[k] = 'hidden'
            elif isinstance(d, list):
                d = [hide_sensitive_info(item) for item in d]
                
            return d
        
        # Make a deep copy of the config to ensure the original is not modified
        safe_config = copy.deepcopy(self.config)
        
        # Traverse and hide sensitive information
        safe_config = hide_sensitive_info(safe_config)

        return safe_config