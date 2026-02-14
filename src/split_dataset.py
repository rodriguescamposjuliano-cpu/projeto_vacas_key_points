"""Script para separar arquivos de dataset em pastas train/test baseado em catalog."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def generate_log_report(
    stats: Dict[str, List[str]],
    log_path: str = "outputs/dataset_split_report.txt",
) -> None:
    """
    Gera relatório em TXT com detalhes da separação.

    Args:
        stats: Dicionário com estatísticas da operação
        log_path: Caminho para salvar o arquivo de log
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE SEPARAÇÃO DE DATASET\n")
        f.write("=" * 80 + "\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Resumo geral
        f.write("RESUMO GERAL\n")
        f.write("-" * 80 + "\n")
        f.write(f"Arquivos Train: {len(stats['train'])}\n")
        f.write(f"Arquivos Test: {len(stats['test'])}\n")
        f.write(f"Arquivos Não Encontrados: {len(stats['not_found'])}\n")
        f.write(f"Total Catálogo: {len(stats['train']) + len(stats['test']) + len(stats['not_found'])}\n")
        f.write(f"Sucesso: {(len(stats['train']) + len(stats['test'])) / (len(stats['train']) + len(stats['test']) + len(stats['not_found'])) * 100:.2f}%\n\n")

        # Arquivos TRAIN
        f.write("ARQUIVOS TRAIN\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total: {len(stats['train'])} arquivos\n\n")
        for filename in sorted(stats["train"]):
            f.write(f"  ✓ {filename}\n")

        f.write("\n")

        # Arquivos TEST
        f.write("ARQUIVOS TEST\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total: {len(stats['test'])} arquivos\n\n")
        for filename in sorted(stats["test"]):
            f.write(f"  ✓ {filename}\n")

        f.write("\n")

        # Arquivos NÃO ENCONTRADOS
        if stats["not_found"]:
            f.write("ARQUIVOS NÃO ENCONTRADOS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total: {len(stats['not_found'])} arquivos\n\n")
            for filename in sorted(stats["not_found"]):
                f.write(f"  ✗ {filename}\n")

    print(f"✓ Relatório salvo em: {log_path}")


def separate_datasets_by_split(
    catalog_path: str = "data/raw/classification_dataset.parquet",
    dataset_source_path: str = "data/raw/dataset_classificação",
    output_base_path: str = "data/raw",
    generate_report: bool = True,
) -> Dict[str, List[str]]:
    """
    Separa arquivos de dataset em pastas train/test baseado no catálogo.

    Args:
        catalog_path: Caminho para o arquivo parquet com o catálogo
        dataset_source_path: Caminho para a pasta com os arquivos originais
        output_base_path: Caminho base para salvar as pastas train/test
        generate_report: Se True, gera relatório em TXT

    Returns:
        Dicionário com contagem de arquivos por split
    """
    # Carregar catálogo
    print(f"\n{'='*80}")
    print("INICIANDO SEPARAÇÃO DE DATASET")
    print(f"{'='*80}\n")
    print(f"Carregando catálogo: {catalog_path}")
    catalog_df = pd.read_parquet(catalog_path)

    print(f"✓ Catálogo carregado com {len(catalog_df)} registros")
    print(f"  Colunas: {list(catalog_df.columns)}")
    print(f"  Splits: {', '.join(catalog_df['split'].unique())}\n")

    source_path = Path(dataset_source_path)
    base_path = Path(output_base_path)

    # Criar pastas de saída
    train_path = base_path / "train"
    test_path = base_path / "test"
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    print(f"Pastas de saída criadas:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}\n")

    stats = {"train": [], "test": [], "not_found": []}
    processed = 0
    errors = 0

    # Separar arquivos
    print(f"Processando {len(catalog_df)} arquivos do catálogo...")
    for idx, (_, row) in enumerate(catalog_df.iterrows(), 1):
        filename = row["filename"]
        split = row["split"].lower()
        cow_id = row.get("cow_id", "unknown")

        # Procurar o arquivo na pasta source dentro da subpasta de cow_id
        source_file = source_path / str(cow_id) / filename

        if not source_file.exists():
            stats["not_found"].append(filename)
            if idx % 100 == 0:
                print(f"  [{idx}/{len(catalog_df)}] ⚠️  {filename} (Vaca: {cow_id})")
            continue

        # Determinar pasta de destino
        if split == "train":
            dest_folder = train_path
            stats["train"].append(filename)
        elif split == "test":
            dest_folder = test_path
            stats["test"].append(filename)
        else:
            print(f"  ✗ Split desconhecido '{split}' para {filename}")
            errors += 1
            continue

        # Criar subpastas se necessário
        dest_file = dest_folder / filename
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # Copiar arquivo
        try:
            shutil.copy2(source_file, dest_file)
            processed += 1
            if processed % 100 == 0:
                print(f"  [{idx}/{len(catalog_df)}] ✓ {processed} arquivos processados")
        except Exception as e:
            print(f"  ✗ Erro ao copiar {filename}: {e}")
            errors += 1

    # Exibir resumo
    print(f"\n{'='*80}")
    print("RESUMO DA SEPARAÇÃO")
    print(f"{'='*80}")
    print(f"✓ Train: {len(stats['train'])} arquivos")
    print(f"✓ Test: {len(stats['test'])} arquivos")
    print(f"⚠️  Não encontrados: {len(stats['not_found'])} arquivos")
    print(f"✗ Erros: {errors}")
    print(f"\nTotal processado com sucesso: {len(stats['train']) + len(stats['test'])} arquivos")
    total = len(stats['train']) + len(stats['test']) + len(stats['not_found'])
    if total > 0:
        success_rate = (len(stats['train']) + len(stats['test'])) / total * 100
        print(f"Taxa de sucesso: {success_rate:.2f}%\n")

    # Gerar relatório
    if generate_report:
        generate_log_report(stats)

    return stats


if __name__ == "__main__":
    separate_datasets_by_split()
