# Identity-Preserving LoRA Style Transfer (IP-LoRA)

📝 **Overview**

**IP-LoRA** (Identity-Preserving LoRA) is a lightweight, modular, and diffusion-based framework for **portrait stylization** that preserves the **identity** of the subject. Built upon the strengths of **B-LoRA** and **ConsisLoRA**, IP-LoRA introduces a novel identity consistency loss derived from pretrained **ArcFace** or **DINOv2** embeddings to ensure semantically faithful stylizations.

This repository contains training scripts, inference pipelines and evaluation tools for reproducing all results in our [final project report](final_report.pdf).

---

✨ **Features**

- Supports **image-to-image** and **text-to-image** stylization modes  
- Plug-and-play **LoRA adapters** for content and style  
- **Identity preservation loss** using ArcFace or DINOv2  
- Evaluated with **CLIP**, **DINOv2**, and **DreamSim** on content and style alignment  
