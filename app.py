"""Shim raíz para Hugging Face Spaces (SDK: Streamlit).

HF Spaces busca app.py en la raíz. Este archivo simplemente
ejecuta la app Streamlit del paquete demo.
"""

import sys
from pathlib import Path

# Fallback: si el paquete no está instalado en modo editable (entornos sin uv),
# añadir src/ al path para que el import funcione igualmente.
_src = Path(__file__).parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from churn_agent.demo.streamlit_app import main  # noqa: E402

main()
