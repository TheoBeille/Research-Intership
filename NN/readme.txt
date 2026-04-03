# **Learned Deviations – Implementation**

## **Overview**

This repository contains implementations of neural network-based approaches to learn deviations in an iterative algorithm. The goal is to improve convergence behavior by incorporating learned components.

The work is structured in two main parts:

- **Only_dev/**: Experiments with limited output features (**only the deviations**)
- **Everything/**: Extended models with more outputs (**work in progress**)

---

## **Project Structure**

### **1. Only_dev/**

This folder contains initial experiments where neural networks are used to learn deviations under simplified settings.

#### **a) Small-scale experiment (16×16 images)**

- **File**: `Only_dev/AlgMLP_onlydev.py`

- **Description**:  
  A neural network is trained on small images (16×16) to learn deviations in the iterative process.

- **Methodology**:
  - The algorithm is unrolled over several iterations
  - The learned deviations are applied at each step
  - Performance is evaluated by comparing convergence against the **zero deviations**

---

#### **b) Primal-only neural network**

- **File**: `Only_dev/AlgMLP_primal.py`

- **Description**:  
  A neural network is applied exclusively to the primal variables.

- **Objective**:
  - Assess whether restricting the model to primal variables is sufficient
  - Compare convergence performance with the full deviation model above

---

#### **c) Scaling experiments**

- **Goal**:  
  Extend the approach trained on small-scale data to larger problem sizes.

- **Challenge**:  
  Generalization of learned deviations across different resolutions and problem dimensions.

---

## **Notes**

- Initial experiments are conducted on small images for computational efficiency
- Results should be interpreted as **proof-of-concept** rather than final performance benchmarks
