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

[Glow](https://github.com/pytorch/glow) is a machine learning compiler from
Facebook that accelerates the performance of deep learning frameworks on
different hardware platforms. In this course we will study how the Glow
framework works, and show how to use quantization to generate code for a model
in a NeuroMorphic Processor (NMP).

[XLA](https://www.tensorflow.org/xla) (Accelerated Linear Algebra) is
a domain-specific compiler for linear algebra, from Google, that can accelerate
TensorFlow models. It leverages on the effcient partitioning of tensor data
across the memory hierarchy of a processor to generate high-performance code.
In this course we will study XLA HLO graph representation, and show how to
generate efficient code for operators like convolution.

[JAX](https://github.com/google/jax) (Just After Execution) is a Python library
and JIT compiler designed by Google to speedup high-performance numerical
computing and machine learning applications. It has been extensively used  by
DeepMind in its projects, and leverages on Google TensorFlow XLA to generate
efficent code for CPUs/GPUs. In this course we will explain JAX in details and
discuss how model graphs can be parallelized in a cluster.

## First Project: ONNX

In 2017, AWS, Microsoft, and Facebook came together to launch the Open Neural
Network Exchange (ONNX), which defines a standard for ML interoperability. ONNX
has two components: a common set of operators and a common file format.
Operators are the building blocks of machine learning and deep learning models.
By standardizing a common set of operators, ONNX makes it easy to consume deep
learning models trained in any of the supported frameworks. It defines an
extensible computation graph model, as well as definitions of built-in
operators and standard data types.

The common file format of ONNX becomes the lowest common denominator to
represent a model. Once a model is exported to ONNX, irrespective of the
framework it is trained in, it exposes a standard graph and set of operators
based on the specification. Every model is converted into a standard
intermediate representation (IR) that is well-defined and well-documented. By
providing a common representation of the computation graph, ONNX helps
developers choose the right framework for their task, allows authors to focus
on innovative enhancements, and enables hardware vendors to streamline
optimizations for their platforms.

Details about this project are in the [ONNX
Section](https://github.com/MO436-MC934/notebooks/wiki/1.ONNX-Model#1open-neural-network-exchange--onnx)
in the Wiki.

## Second Project: The GLOW Plataform

Glow is a machine learning compiler and execution engine for hardware
accelerators. It is designed to be used as a backend for high-level machine
learning frameworks. The compiler is designed to allow state of the art
compiler optimizations and code generation of neural network graphs.

### How does it work?

Glow lowers a traditional neural network dataflow graph into a two-phase
strongly-typed [intermediate
representation](https://github.com/pytorch/glow/blob/master/docs/IR.md) (IR).
The high-level IR allows the optimizer to perform domain-specific
optimizations. The lower-level instruction-based address-only IR allows the
compiler to perform memory-related optimizations, such as instruction
scheduling, static memory allocation and copy elimination. At the lowest level,
the optimizer performs machine-specific code generation to take advantage of
specialized hardware features. Glow features a lowering phase which enables the
compiler to support a high number of input operators as well as a large number
of hardware targets by eliminating the need to implement all operators on all
targets. The lowering phase is designed to reduce the input space and allow new
hardware backends to focus on a small number of linear algebra primitives. The
design philosophy is described in an [arXiv](https://arxiv.org/abs/1805.00907)
paper.

Details about this project are in the [Glow
Section](https://github.com/MO436-MC934/notebooks/wiki/2.Glow-PLatform#2-glow-platform)
in the Wiki.

## Third Project: GEMM Optimization

THe third project requires you to generate the ONNX model for ResNet, and fill
in table with rows (C1, C2, C3 and C4)  and columns a list of four convolutions
(no-BLAS, BLAS). Measure the model performance for each entry in the table.

Details about this project are in the [GEMM Section]() in the Wiki.

## Fourth Project: ML Code Optimization

The fourth project requires you to generate the ONNX model for ResNet, and fill
in table with rows being 4 sets of triples (k1,k2,k3) and columns (WS,IS).
Measure the model performance for each entry in the table.

Details about this project are in the [ML Code Section]() in the Wiki.

## Fifth project: The JAX Library

In the fifth project, you receive the ResNet model in python. The projerct
requires you to measure the performance of each type of operator, with and
without JAX, filling in a table where rows are (Model operator) and columns are
(No-JAX,JAX).

Details about this project are in the [JAX Library]() in the Wiki.

