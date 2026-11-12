scMAGZ
Overview
scMAGZ is a graph-based framework for integrating paired single-cell multimodal data. It combines modality-specific graph learning with an attention-based fusion mechanism to generate joint cell embeddings while encouraging the preservation of local neighborhood structures within each modality.

The integrated embeddings can support downstream analyses, including cell clustering, visualization, and trajectory inference. The framework also includes RNA reconstruction and selective imputation components for downstream expression analysis.
Supported Data Types
scMAGZ is primarily designed for paired scRNA-seq and scATAC-seq data. It can also be adapted to paired RNA and surface protein measurements from CITE-seq.
- RNA–ATAC: Integrates gene expression with chromatin accessibility features.
- RNA–ADT: Integrates gene expression with antibody-derived tag (ADT) measurements of surface protein abundance. This extension uses an ADT-specific reconstruction objective.
The two modalities must contain measurements from the same cells, with cell identifiers matched and ordered consistently before analysis.
Outputs and Applications
Depending on the analysis workflow, scMAGZ provides integrated cell embeddings, reconstructed RNA expression, and selectively imputed RNA expression. These outputs can be used for cell population identification, multimodal visualization, and downstream biological analyses.
