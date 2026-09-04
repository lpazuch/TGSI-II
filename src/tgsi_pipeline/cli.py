from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tgsi-pipeline",
        description="Coleta e consolida variaveis independentes para o projeto TGSI.",
    )
    parser.add_argument("--config", required=True, help="Caminho do arquivo JSON de configuracao.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida a configuracao e escreve apenas o relatorio da execucao.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Interrompe na primeira falha de fonte, em vez de gerar saida parcial.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Imprime a configuracao carregada antes de executar o pipeline.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    config_dict = asdict(config)

    if args.print_config:
        print(json.dumps(config_dict, default=str, indent=2, ensure_ascii=False))

    output_base_dir = Path(config.output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    (output_base_dir / "run_config.json").write_text(
        json.dumps(config_dict, default=str, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result = run_pipeline(
        config,
        allow_partial=not args.strict,
        dry_run=args.dry_run,
    )

    print(
        json.dumps(
            {
                "raw_files": result.raw_files,
                "processed_files": result.processed_files,
                "warnings": result.warnings,
                "errors": result.errors,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
