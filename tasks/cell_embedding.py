import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import CellFM
from CellFM_tokenizer import CellFMTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CellFM CLS cell embeddings.")

    parser.add_argument("--input", required=True, help="Input .h5ad file.")
    parser.add_argument("--output", required=True, help="Output .npy file.")
    parser.add_argument("--model", default="CellFM-80M", help="Model name under assets/.")
    parser.add_argument("--assets-dir", default="assets", help="Assets directory.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--filter-cells", action="store_true")
    parser.add_argument("--no-compile", action="store_true")

    return parser.parse_args()


@torch.inference_mode()
def encode_cell_embeddings(model, expr, gene, zero_idx, batch_size, device):
    model.eval()
    outputs = []

    n_cells = expr.shape[0]

    gene_device = gene.to(device, non_blocking=True) if gene.ndim == 1 else None
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)

        expr_b = expr[start:end].to(device, non_blocking=True)
        # gene_b = gene[start:end].to(device, non_blocking=True)
        if gene.ndim == 1:
            gene_b = gene_device
        else:
            gene_b = gene[start:end].to(device, non_blocking=True)
        zero_b = zero_idx[start:end].to(device, non_blocking=True)

        # emb = model.encode_cells(expr_b, gene_b, zero_b)
        # outputs.append(emb.cpu())
        #Thêm AMP quanh forward
        use_amp = device.type == "cuda"

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            emb = model.encode_cells(expr_b, gene_b, zero_b)
        
        outputs.append(emb.float().cpu())

    return torch.cat(outputs, dim=0).numpy()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    model = CellFM.from_pretrained(
        model_name=args.model,
        assets_dir=args.assets_dir,
        map_location=device,
    ).to(device)
    
    if device.type == "cuda" and not args.no_compile:
        model = torch.compile(model)

    #tokenizer
    tokenizer = CellFMTokenizer.from_pretrained(
        model_name=args.model,
        assets_dir=args.assets_dir,
    )
    
    # expr, gene, zero_idx = tokenizer.encode(args.input)
    expr, gene, zero_idx = tokenizer.encode(
        args.input,
        filter_cells=args.filter_cells,
    )
    if gene.ndim == 2 and torch.equal(gene[0], gene[-1]):
        gene = gene[0]
        
    #Pin memory sau tokenizer
    if device.type == "cuda":
        expr = expr.pin_memory()
        gene = gene.pin_memory()
        zero_idx = zero_idx.pin_memory()

    embeddings = encode_cell_embeddings(
        model=model,
        expr=expr,
        gene=gene,
        zero_idx=zero_idx,
        batch_size=args.batch_size,
        device=device,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)

    print(f"Saved embeddings: {output}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()