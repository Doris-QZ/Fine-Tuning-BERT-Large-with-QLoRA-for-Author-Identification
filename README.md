# Fine-Tuning BERT Large with QLoRA for Author Identification 🧠 

This repository explores **parameter-efficient fine-tuning (PEFT)** of large language models for an **Author Identification** task using the [Spooky Author Identification Dataset](https://www.kaggle.com/competitions/spooky-author-identification) from Kaggle. The primary challenge of this task lies in the fact that the excerpts come from the **same literary genre**--where thematic content and vocabulary often overlap across authors--making it difficult to distinguish between writing styles.

Unlike [my previous experiments](https://github.com/Doris-QZ/spooky_author_identification/blob/main/3_BERT_Spooky_Author_Identification.ipynb) where only the last few layers of **BERT Base** were fine-tuned (achieving a validation loss of 0.45 and accuracy of 0.83), this project investigates how **QLoRA (Quantized Low-Rank Adaptation)** can be applied to fine-tune the **BERT Large** model efficiently, given limited computational resources (one **A100 GPU** from Google Colab) and a relatively small dataset (**fewer than 20k samples**).

***

### Project Goals 🎯

* Explore **QLoRA** as a memory-efficient fine-tuning approach for large transformer models.
* Evaluate the effect of **different LoRA configurations**:
  * **Rank** and **alpha** settings (e.g., r={8, 16}; a={2, 4, 8, 12})
  * Different **target modules** (e.g., Q, K, V | All attention | All Linear)
* Compare model performance using validation metrics (loss and accuracy).
* Combine the best-performing fine-tuned models through **ensemble learning** to improve test predictions.

### Repository Structure 🧩
* The first two notebooks--`QLoRA_r8a4_AllLin_Author_Identification.ipynb` and `QLoRA_r16a8_QKV_Author_Identification.ipynb`--fine-tune the BERT Large model using the same procedure but with different hyperparameter settings.
* The third notebook, `Ensemble_Results.ipynb`, ensembles the predictions from the best two models (shown in the first two notebooks) to produce the final results.

### Key Results ⚙️

* **Best Single Model**: r=8, α=4, target_modules=[All Linear]
  * Validation Loss: **0.36**

  * Validation Accuracy: **0.87**

* **Ensemble Model**: r8a4_AllLin + r16a8_QKV
  * Validation Loss: **0.32**

  * Validation Accuracy: **0.86**
    
  * Test Loss: **0.33**

### Summary 🚀

This project demonstrates that:

* Even with fewer than **20k samples** and only **one A100 GPU**, **BERT Large** can be effectively fine-tuned using **QLoRA**.
* **Ensemble learning** enhances both stability and performance across validation and test sets.
* **Parameter-efficient fine-tuning** bridges the gap between resource constraints and large model capabilities — making **LLM adaptation** more accessible to smaller teams and individual practitioners.
