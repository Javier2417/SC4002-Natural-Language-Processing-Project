
# NTU SC4002: Search Intent Classification (Group 54)

> **Master Data Pipeline & Advanced Recurrent Architectures**

## 📌 Project Overview

This repository contains the end-to-end NLP pipeline for the **TREC Question Classification** task. The objective is to categorize user queries into six semantic topics (ABBR, DESC, ENTY, HUM, LOC, NUM). This project was a collaborative effort across three specialized sub-teams to benchmark RNN, CNN, and Advanced RNN architectures.

## 🛠️ Team 3 Deliverables (Aaron & Javier)

As part of Team 3, we were responsible for the foundational data infrastructure and the implementation of high-performance recurrent baselines.

### 1. Master Data Pipeline (Part 0 & 1)

We developed a standardized pipeline to ensure consistency across all experimental groups:

* **OOV Mitigation Strategy**: Implemented a pre-loaded embedding matrix utilizing **GloVe.6B.300d**. Words not present in the pre-trained dictionary were initialized via **Normal Distribution**, allowing the model to learn representations for unknown tokens during training.
* **Semantic Analysis**: Conducted **t-SNE dimensionality reduction** on the top 20 frequent words per topic to visualize cluster separation and identify potential classification bottlenecks (e.g., overlap between DESC and NUM).

### 2. Recurrent Model Ownership (Javier Tin)

I personally took ownership of the design, implementation, and optimization of the core recurrent architectures:

* **BiGRU & BiLSTM**: Architected the bidirectional recurrent layers to capture both forward and backward temporal dependencies in the question sequences.
* **Sentence Representation**: Benchmarked Mean Pooling, Max Pooling, and Global Pooling, determining that **Mean Pooling** provided the most stable generalized summary for this dataset.
* **Optimization**: Conducted a refined grid search across **108 combinations**, tuning hidden dimensions (256/384), layer counts, and dropout rates to achieve peak performance.

## 🚀 Advanced Enhancements

To push performance beyond the baseline, we implemented two critical improvements:

* **Attention Mechanism**: Integrated a context-aware Attention layer with `tanh` activation to dynamically weight specific "clue words" in queries, overcoming the information bottleneck of traditional RNNs.
* **Focal Loss Implementation**: Addressed extreme class imbalance (specifically the rarity of ABBR and ENTY topics) by applying **Focal Loss ()**. This strategy successfully boosted overall test accuracy to **90.11%**.

## 📊 Performance Summary

| Model Configuration | Overall Test Accuracy |
| --- | --- |
| **RNN Baseline (L2 Regularization)** | 88.00% |
| **BiGRU (Tuned)** | 89.00% |
| **BiLSTM + Focal Loss** | **90.11%** |

## 👥 Contribution List (Group 54)

| Member | Team | Primary Contributions |
| --- | --- | --- |
| **Tin Jing Lun Javier** | **3** | **Lead for BiGRU/BiLSTM Implementation, Focal Loss & Attention Logic** |
| Chen Guan Zong Aaron | 3 | Data Pipeline (Part 0/1), OOV Strategy, Embedding Analysis |
| Wei Hong | 1 | RNN Architecture Exploration & Benchmarking |
| Aryan Nangia | 1 | RNN Regularization Strategies & Validation |
| Avanesh | 2 | CNN Architecture Design & Feature Map Analysis |
| Jayden Yeo He | 2 | CNN Hyperparameter Tuning & Signal Extraction |

