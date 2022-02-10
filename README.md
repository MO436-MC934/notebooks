# MO436-MC934 -- Machine Learning under the hood

Machine Learning (ML) has become a central area in modern computing. The number
of applications using ML models has grown exponentially, and as such the demand
for engineers who know how to pilot  toolchains  like PyTorch, TensorFlow, etc.
On the other hand, there is much more interesting knowleged in ML than just
learning how to apply it to a problem. Computing a ML model graph is
a computational intensive task that requires a number of optimization and
parallelization algorithms that are hidden from the typical user. Understanding
these algorithms is a central tool for the  modern data-scientist,  as it
differentiates the professional from the crowd of people working in this area.

In this course we will cover the most important algorithms for paralleization
and optimization that are used to compute  ML graphs. This will be done by
looking under the hood of three major ML Engines: (a) Facebook Glow; (b) Google
TensorFLow XLA; and (c) Google JAX.


## First Project: ONNX

In the first project, you receive a template in python of LeNet Model. You will
complement the model, generate the ONNX graph and use Netron tool to display
and print the model to pdf. Details about this project are in the
[First Project][P1] notebook.

## Second Project: The GLOW Plataform

The second project requires you to read the ONNX graph from previous lab,
generate the HIR as is, do fusion and quantization and generate new HIR.
Details about this project are in the [Second Project][P2] notebook.

## Third Project: GEMM Optimization

THe third project requires you to generate the ONNX model for ResNet, and fill
in table with rows (C1, C2, C3 and C4)  and columns a list of four convolutions
(no-BLAS, BLAS). Measure the model performance for each entry in the table.
Details about this project are in the [Third Project][P3] notebook.

## Fourth Project: ML Code Optimization

The fourth project requires you to generate the ONNX model for ResNet, and fill
in table with rows being 4 sets of triples (k1,k2,k3) and columns (WS,IS).
Measure the model performance for each entry in the table. Details about this
project are in the [Fourth Project][P4] notebook.

## Fifth project: The JAX Library

In the fifth project, you receive the ResNet model in python. The projerct
requires you to measure the performance of each type of operator, with and
without JAX, filling in a table where rows are (Model operator) and columns are
(No-JAX,JAX). Details about this project are in the [Fifth Project][P5]
notebook.

