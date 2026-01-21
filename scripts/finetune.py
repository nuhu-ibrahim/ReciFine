from __future__ import annotations

from train import main
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    main(caller="finetune")
