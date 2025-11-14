# Multi-level Mixture of Experts (MMoE) for Multimodal Entity Linking

This repository contains an implementation of the **Multi-level Mixture of Experts (MMoE)** model for the **Multimodal Entity Linking (MEL)** task.  
The architecture is based on the research paper **“Multi-level Mixture of Experts for Multimodal Entity Linking (KDD 2025)”**, and improves MEL performance by addressing mention ambiguity and dynamic multimodal feature selection.

---

## 🔍 Overview

Multimodal Entity Linking aims to link a textual mention (optionally with an accompanying image) to the correct entity in a knowledge graph such as WikiData.

However, MEL is challenging due to:

1. **Mention Ambiguity** – Short or unclear mention contexts create confusion.  
2. **Dynamic Selection of Modal Content** – Not all tokens or image regions contribute equally.

The **MMoE model** resolves these challenges using:

- **Description-aware Mention Enhancement (DME)**  
  Enhances ambiguous mentions using WikiData descriptions chosen by an LLM (e.g., LLaMA).

- **Multimodal Feature Extraction (MFE)**  
  Uses CLIP to extract coarse- and fine-grained text & image embeddings.

- **Intra-level MoE (IntraMoE)**  
  Learns within-modality (text-only, image-only) dynamic feature importance.

- **Inter-level MoE (InterMoE)**  
  Learns cross-modal (text ↔ image) interactions.

The final matching score is computed using both intra-modal and cross-modal similarity.

---

## 📦 Dependencies

To create the environment:

```bash
conda create -n mmoe python=3.7 -y
conda activate mmoe
pip install torch==1.11.0+cu113 \
            transformers==4.27.1 \
            torchmetrics==0.11.0 \
            tokenizers==0.12.1 \
            pytorch-lightning==1.7.7 \
            omegaconf==2.2.3 \
            pillow==9.3.0

## Project Structure

MEL-MMoE/
│── config/                 # YAML configuration files
│── data/                   # Dataset folders
│── codes/
│   ├── model/
│   │   ├── encoder.py      # CLIP-based encoder
│   │   ├── moe.py          # Switch-MoE
│   │   ├── mmoe.py         # MMoE architecture
│   ├── dataset.py          # Dataset loading logic
│   ├── train.py            # Training
│   ├── evaluate.py         # Evaluation
│   ├── predict.py          # Inference
│── logs/                   # Training logs
│── main.py                 # Entry point script
│── requirements.txt
│── README.md
