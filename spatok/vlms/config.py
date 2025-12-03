"""
Configuration management for VLM API keys.

Supports multiple methods for API key storage:
1. .env file in project root (recommended)
2. Environment variables
3. Config file in home directory (~/.vlm_config.json)
4. Direct parameter passing
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict


class VLMConfig:
    """Manages API keys for VLM services."""
    
    # Default config file location
    DEFAULT_CONFIG_PATH = Path.home() / '.vlm_config.json'
    
    # Environment variable names
    ENV_VARS = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'xai': 'XAI_API_KEY',
        'reka': 'REKA_API_KEY',
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize VLM configuration.
        
        Args:
            config_path: Optional path to config JSON file
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._keys: Dict[str, str] = {}
        self._load_config()
    
    def _load_config(self):
        """Load API keys from various sources (priority order)."""
        # 1. Try loading from .env file in project root
        self._load_from_dotenv()
        
        # 2. Try loading from environment variables
        self._load_from_env()
        
        # 3. Try loading from config file
        self._load_from_file()
    
    def _load_from_dotenv(self):
        """Load from .env file using python-dotenv if available."""
        try:
            from dotenv import load_dotenv
            # Look for .env in project root (2 levels up from this file)
            env_path = Path(__file__).parent.parent.parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            pass  # python-dotenv not installed
    
    def _load_from_env(self):
        """Load API keys from environment variables."""
        for service, env_var in self.ENV_VARS.items():
            value = os.getenv(env_var)
            if value and service not in self._keys:
                self._keys[service] = value
    
    def _load_from_file(self):
        """Load API keys from JSON config file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Map file keys to service names
                key_mapping = {
                    'openai_api_key': 'openai',
                    'anthropic_api_key': 'anthropic',
                    'google_api_key': 'google',
                    'xai_api_key': 'xai',
                    'reka_api_key': 'reka',
                }
                
                for file_key, service in key_mapping.items():
                    if file_key in file_config and service not in self._keys:
                        self._keys[service] = file_config[file_key]
            except (json.JSONDecodeError, IOError):
                pass  # Config file invalid or unreadable
    
    def get(self, service: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get API key for a service.
        
        Args:
            service: Service name ('openai', 'anthropic', 'google', 'xai', 'reka')
            default: Default value if key not found
            
        Returns:
            API key string or None
        """
        return self._keys.get(service.lower(), default)
    
    def get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.get('openai')
    
    def get_anthropic_key(self) -> Optional[str]:
        """Get Anthropic API key."""
        return self.get('anthropic')
    
    def get_google_key(self) -> Optional[str]:
        """Get Google API key."""
        return self.get('google')
    
    def get_xai_key(self) -> Optional[str]:
        """Get X.AI API key."""
        return self.get('xai')
    
    def get_reka_key(self) -> Optional[str]:
        """Get Reka API key."""
        return self.get('reka')
    
    def set(self, service: str, api_key: str):
        """
        Set API key for a service (runtime only, not persisted).
        
        Args:
            service: Service name
            api_key: API key string
        """
        self._keys[service.lower()] = api_key
    
    def save_to_file(self, path: Optional[Path] = None):
        """
        Save current API keys to JSON config file.
        
        Args:
            path: Optional custom path (default: ~/.vlm_config.json)
        """
        save_path = path or self.config_path
        
        config_data = {
            f"{service}_api_key": key
            for service, key in self._keys.items()
        }
        
        with open(save_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Set restrictive permissions (owner read/write only)
        save_path.chmod(0o600)
    
    def create_template_env(self, path: Optional[Path] = None):
        """
        Create a template .env file.
        
        Args:
            path: Path to create .env file (default: project root)
        """
        if path is None:
            path = Path(__file__).parent.parent.parent / '.env'
        
        template = """# VLM API Keys Configuration
# Add your API keys below and keep this file secure (already in .gitignore)

# OpenAI API Key (for GPT-4V, GPT-4o)
OPENAI_API_KEY=sk-your-key-here

# Anthropic API Key (for Claude)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Google API Key (for Gemini)
GOOGLE_API_KEY=your-key-here

# X.AI API Key (for Grok)
XAI_API_KEY=xai-your-key-here

# Reka AI API Key (optional)
REKA_API_KEY=your-key-here
"""
        
        with open(path, 'w') as f:
            f.write(template)
        
        path.chmod(0o600)
        print(f"Created template .env file at: {path}")
    
    def __repr__(self) -> str:
        """String representation showing which keys are configured."""
        configured = [service for service in self.ENV_VARS.keys() if service in self._keys]
        return f"VLMConfig(configured: {configured})"


# Global config instance
_global_config: Optional[VLMConfig] = None


def get_config() -> VLMConfig:
    """
    Get global VLM configuration instance.
    
    Returns:
        VLMConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = VLMConfig()
    return _global_config


def get_api_key(service: str) -> Optional[str]:
    """
    Convenience function to get API key for a service.
    
    Args:
        service: Service name ('openai', 'anthropic', 'google', 'xai', 'reka')
        
    Returns:
        API key or None
    """
    return get_config().get(service)
