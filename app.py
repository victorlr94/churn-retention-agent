"""Shim raíz para Hugging Face Spaces (SDK: Streamlit).

HF Spaces busca app.py en la raíz. Este archivo simplemente
ejecuta la app Streamlit del paquete demo.
"""

from churn_agent.demo.streamlit_app import main

main()
