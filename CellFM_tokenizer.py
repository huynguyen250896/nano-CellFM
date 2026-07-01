import csv
from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np
import scanpy as sc
import torch
from scipy import sparse


class CellFMTokenizer:
    def __init__(
        self,
        gene_info_path: Union[str, Path],
        pad_len: int = 2048,
        min_genes: int = 200,
        max_genes: int = 2048,
        normalize_scale: float = 1e5,
    ):
        self.gene_info_path = Path(gene_info_path)
        self.pad_len = pad_len
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.normalize_scale = normalize_scale

        self.genes = self._load_genes(self.gene_info_path)
        self.gene_to_id = {gene: i + 1 for i, gene in enumerate(self.genes)}
        self.n_genes = len(self.genes)

    @staticmethod
    def _load_genes(path: Path):
        genes = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            for row in reader:
                if not row:
                    continue
                genes.append(row[0])

        if len(genes) == 0:
            raise ValueError(f"No genes found in {path}")

        return genes

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "CellFM-80M",
        assets_dir: str = "assets",
        **kwargs,
    ):
        model_dir = Path(assets_dir) / model_name
        gene_info_path = model_dir / "expand_gene_info.csv"

        if not gene_info_path.exists():
            raise FileNotFoundError(
                f"Could not find expand_gene_info.csv at {gene_info_path}. "
                f"Expected layout: assets/{model_name}/expand_gene_info.csv"
            )

        return cls(gene_info_path=gene_info_path, **kwargs)

    def encode_adata(
        self,
        adata,
        filter_cells: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if adata.n_vars < self.min_genes:
            raise ValueError(
                f"Only {adata.n_vars} genes found; "
                f"expected at least {self.min_genes}."
            )
    
        if filter_cells:
            sc.pp.filter_cells(adata, min_genes=self.min_genes)
    
        x = adata.X
        if sparse.issparse(x):
            x = x.toarray()
        else:
            x = np.asarray(x)
    
        x = x.astype(np.float32)
    
        gene_ids = np.array(
            [
                self.gene_to_id.get(str(g), 0)
                for g in adata.var_names.astype(str)
            ],
            dtype=np.int64,
        )
    
        # gene = np.repeat(gene_ids[None, :], x.shape[0], axis=0)
        gene = gene_ids
        
        zero_idx = np.concatenate(
            [
                np.ones((x.shape[0], 1), dtype=np.float32),
                (x != 0).astype(np.float32),
            ],
            axis=1,
        )
    
        return (
            torch.from_numpy(x),
            torch.from_numpy(gene),
            torch.from_numpy(zero_idx),
        )

    def encode(
        self,
        input_data: Union[str, Path, object],
        filter_cells: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(input_data, (str, Path)):
            adata = sc.read_h5ad(str(input_data))
        else:
            adata = input_data
    
        return self.encode_adata(
            adata,
            filter_cells=filter_cells,
        )


def get_pretrained_tokenizer(
    model_name: str = "CellFM-80M",
    assets_dir: str = "assets",
) -> CellFMTokenizer:
    return CellFMTokenizer.from_pretrained(
        model_name=model_name,
        assets_dir=assets_dir,
    )