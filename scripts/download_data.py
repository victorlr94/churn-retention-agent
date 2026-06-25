"""Descarga y verificación del dataset IBM Telco Customer Churn.

Dado que no usamos la Kaggle API, este script:
1. Imprime instrucciones exactas de descarga manual.
2. Si el CSV ya existe en data/raw/, verifica su integridad por SHA-256.
3. Si el hash coincide con el valor versionado en data/checksums.txt, informa OK.
4. Si el CSV es nuevo (no hay checksum guardado), computa y guarda el hash
   para que el equipo lo versione.

Uso:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --verify-only   # solo comprueba; no guarda
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHECKSUMS_FILE = PROJECT_ROOT / "data" / "checksums.txt"

# Nombre esperado del archivo tras la descarga.
DATASET_FILENAME = "telco_customer_churn.csv"

DOWNLOAD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════╗
║         IBM Telco Customer Churn — Descarga manual                      ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. Ve a Kaggle y busca:                                                 ║
║     "IBM Telco Customer Churn" (versión rica, ~7 043 filas)              ║
║     Dataset de referencia: blastchar/telco-customer-churn                ║
║     (o la versión extendida con CLTV, Churn Score, Satisfaction Score)   ║
║                                                                          ║
║  2. Descarga el CSV principal.                                           ║
║                                                                          ║
║  3. Renómbralo a:                                                        ║
║       telco_customer_churn.csv                                           ║
║                                                                          ║
║  4. Colócalo en:                                                         ║
║       data/raw/telco_customer_churn.csv                                  ║
║                                                                          ║
║  5. Vuelve a correr este script para verificar la integridad.            ║
║                                                                          ║
║  NOTA: data/raw/ está en .gitignore — el CSV nunca entra al repo.        ║
║        Solo el hash (checksums.txt) se versiona.                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums() -> dict[str, str]:
    if not CHECKSUMS_FILE.exists():
        return {}
    result: dict[str, str] = {}
    for line in CHECKSUMS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":", 1)
        if len(parts) == 2:
            result[parts[0].strip()] = parts[1].strip()
    return result


def save_checksum(filename: str, digest: str) -> None:
    checksums = load_checksums()
    checksums[filename] = digest
    lines = [
        "# SHA-256 checksums de los archivos de data/raw/",
        "# Formato: filename: sha256hash",
        "",
    ]
    for name, h in sorted(checksums.items()):
        lines.append(f"{name}: {h}")
    CHECKSUMS_FILE.write_text("\n".join(lines) + "\n")


def main(verify_only: bool = False) -> int:
    csv_path = RAW_DIR / DATASET_FILENAME

    if not csv_path.exists():
        print(DOWNLOAD_INSTRUCTIONS)
        print(f"[ERROR] No encontrado: {csv_path}")
        return 1

    print(f"[OK] Archivo encontrado: {csv_path}")
    print("     Calculando SHA-256...")
    digest = sha256(csv_path)
    print(f"     SHA-256: {digest}")

    checksums = load_checksums()
    stored = checksums.get(DATASET_FILENAME)

    if stored is None:
        if verify_only:
            print(
                "[WARN] No hay checksum versionado. Corre sin --verify-only para guardarlo."
            )
            return 0
        save_checksum(DATASET_FILENAME, digest)
        print(f"[OK] Checksum guardado en {CHECKSUMS_FILE}")
        print("     Commitea data/checksums.txt para versionarlo.")
        return 0

    if digest == stored:
        print(
            "[OK] Integridad verificada — el archivo coincide con el checksum versionado."
        )
        return 0

    print("[ERROR] Checksum no coincide.")
    print(f"        Esperado: {stored}")
    print(f"        Obtenido: {digest}")
    print(
        "        El archivo puede estar corrupto o ser una versión diferente del dataset."
    )
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verifica la integridad del dataset de Telco Churn."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Solo verifica; no guarda checksum nuevo.",
    )
    args = parser.parse_args()
    sys.exit(main(verify_only=args.verify_only))
