**Disclaimer**: For research only; not for clinical use. Follow providers' ethical terms for models/datasets [1-12].

**GEMMA 3NCORE** (ENCORE, with "3n" as a tribute to Gemma 3n) [1] is a proof-of-concept protoype by Shehab Anwer for The Adimension [2]. It demonstrates **intra-Gemmaverse knowledge transfer** through **cross-architecture knowledge distillation (KD)**, shifting medical imaging knowledge from **MedGemma's [3] MedSigLIP vision encoder** to **Gemma 3n's TIMM vision backbone** [4]. This leverages Gemma 3n's multimodality, especially its **Matryoshka-model nested transformer** for elastic efficient submodel extraction [5] to support scalable, resources- and device-optimised healthcare applications, like in cardiac imaging analysis as in this pilot.

**3NCORE KD pipeline** targets regression tasks for cardiac metrics—End-Diastolic Volume (EDV), End-Systolic Volume (ESV), and Ejection Fraction (EF)—using datasets like EchoNet-Dynamic [6] and CAMUS [7]. Key techniques:
- **Mean Squared Error (MSE)** loss for task-specific predictions.
- Feature matching loss for alignment, with projectors to match layer dimensions and adaptive pooling for memory efficiency.
- **Parameter-Efficient Fine-Tuning (PEFT) through Low-Rank Adaptation (LoRA)** [8] or **Quantised LoRA (QLoRA)** [9] for scalability and resource adaptation.
- Gradient accumulation and curriculum learning through gradual adjustment of feature loss weight for low-resource setups, like ephemral environments (Google Colab / Kaggle notebooks).

As part of the **ATRIOM (Artifact Transformation & Resources Interoperability in Machine Learning) collection**—inspired by atrial functions (Atrium) for interoperability (IO)—ATRIOM has three phases:
- **Reservoir**: Gather artifacts and set up environments, with fresh package installs and automated reporting (JSON/DataFrame) for libraries like PyTorch [10] and Hugging Face Transformers [11].
- **Conduit**: Build data pipelines and bridges, including on-the-fly video frame extraction (e.g., ES/ED frames from EchoNet videos) to save memory, auto GPU/CPU detection with 8-bit quantization [9], and IPython widgets for hyperparameter tuning.
- **Active**: Train, analyze, and generate adapters. The custom student model fuses ES/ED vision features via a sequential layer before regression heads; it supports timestamped checkpointing of LoRA adapters and full models for versioning.

Focused on healthcare, the methodology aligns ethically with **Health AI Developer Foundations** [12] and the **DEITY Principles Framework** [2],  foundational to the Adimension solutions: **D**ata for transparent, diverse inputs; **E**thics for governance; **I**nformatics for interpretable outputs; **T**echnology for adaptive, empowering **Y**ou, both the human & machine, through solutions that bridges human ingenuity with machine intelligence. Altogether, aiming at an impact on **Healthcare and beyond**, through scalable, equitable AI that tackles interoperability and resource constraints [2], as well as a deeper Gemmaverse insights [1,3], boosting computational efficiency to sparks cross-modal innovations [12].

---

**Disclaimer**: For research only; not for clinical use. Follow providers' ethical terms for models/datasets [1-12].

---

### **References**
1. Gemma Team. (2025). Gemma 3n. Google DeepMind. https://ai.google.dev/gemma/docs/gemma-3n.
2. Anwer, S. (2025). The Adimension: Bridging interoperability through DEITY Principles. *European Heart Journal - Imaging Methods and Practice*. https://doi.org/10.1093/ehjimp/qyaf038.
3. Sellergren, A., et al. (2025). MedGemma Technical Report. arXiv:2507.05201.
4. Wightman, R. (2019). PyTorch Image Models. https://github.com/huggingface/pytorch-image-models.
5. Devvrit et al. (2023). MatFormer: Nested Transformer for Elastic Inference. arXiv:2310.07707.
6. Ouyang, D., et al. (2020). Video-based AI for beat-to-beat assessment of cardiac function. *Nature*. https://doi.org/10.1038/s41586-020-2145-8 (EchoNet-Dynamic Dataset).
7. Leclerc, S., et al. (2019). Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography. *IEEE Transactions on Medical Imaging*. https://doi.org/10.1109/TMI.2019.2900516 (CAMUS Dataset).
8. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
9. Dettmers, T., et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. arXiv:2208.07339.
10. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.
11. Wolf, T., et al. (2019). Hugging Face's Transformers. arXiv:1910.03771.
12. Google Health AI Team. (2024). Health AI Developer Foundations. arXiv:2411.15128.