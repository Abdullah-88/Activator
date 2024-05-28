# Activator

GLU Activations as The Core Functions of a Vision Transformer 

Abdullah Nazhat Abdullah, Tarkan Aydin

# Abstract
Transformer architecture currently represents the main driver behind many successes in a variety of tasks addressed by deep learning, especially the recent advances in natural language processing (NLP) culminating with large language models (LLM). In addition, transformer architecture has found a wide spread of interest from computer vision (CV) researchers and practitioners, allowing for many advancements in vision-related tasks and opening the door for multi-task and multi-modal deep learning architectures that share the same principle of operation. One drawback to these architectures is their reliance on the scaled dot product attention mechanism with the softmax activation function, which is computationally expensive and requires large compute capabilities both for training and inference. This paper investigates substituting the attention mechanism usually adopted for transformer architecture with an architecture incorporating gated linear unit (GLU) activation within a multi-layer perceptron (MLP) structure in conjunction with the default MLP incorporated in the traditional transformer design. Another step forward taken by this paper is to eliminate the second non-gated MLP to further reduce the computational cost. Experimental assessments conducted by this research show that both proposed modifications and reductions offer competitive performance in relation to baseline architectures, in support of the aims of this work in establishing a more efficient yet capable alternative to the traditional attention mechanism as the core component in designing transformer architectures.

Paper Link : https://arxiv.org/abs/2405.15953
