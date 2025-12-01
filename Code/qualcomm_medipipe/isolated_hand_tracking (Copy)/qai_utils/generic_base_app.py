# ---------------------------------------------------------------------
# Isolated generic_base_app.py
# ---------------------------------------------------------------------

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from typing import Any

# Simple type alias
RUN_MODEL_RETURN_TYPE = list[torch.Tensor] | torch.Tensor | Any

class BaseCollectionApp(ABC):
    """
    Minimal base class for isolation. 
    Removes need for qai_hub_models.utils.base_model
    """
    
    @abstractmethod
    def run_model(
        self, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> tuple[RUN_MODEL_RETURN_TYPE, ...] | RUN_MODEL_RETURN_TYPE:
        pass

    @classmethod
    def from_pretrained(cls, model: Any) -> BaseCollectionApp:
        """
        Placeholder to satisfy abstract method requirement.
        In our isolated app, we instantiate manually, so this isn't strictly used.
        """
        raise NotImplementedError("Use direct constructor for isolated app.")