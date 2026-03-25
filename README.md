# Awesome-Weight-Space-Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> [!TIP]
> A curated survey-style repository for papers, codebases, and benchmarks on weight space learning.
This repo will be continuously updated. Don't forget to star it and keep tuned!

[![arXiv](https://img.shields.io/badge/arXiv-2603.10090-b31b1b.svg)](https://arxiv.org/abs/2603.10090)
[![GitHub stars](https://img.shields.io/github/stars/Zehong-Wang/Awesome-Weight-Space-Learning?style=social)](https://github.com/Zehong-Wang/Awesome-Weight-Space-Learning)
[![Last Commit](https://img.shields.io/github/last-commit/Zehong-Wang/Awesome-Weight-Space-Learning)](https://github.com/Zehong-Wang/Awesome-Weight-Space-Learning)



## Weight Space Learning

Weight Space Learning is a research perspective that shifts focus from studying neural networks only through their input–output functions to directly analyzing and leveraging their parameters. Unlike conventional training, which treats weights merely as optimization variables, weight space learning regards them as a meaningful domain of study and operation. Existing works in this area can be organized along three complementary dimensions: **(1) weight space understanding**, which investigates the geometry, symmetry, and statistical properties of weights; **(2) weight space discrimination**, which treats weights as a modality for tasks such as embedding, retrieval, and behavior prediction; and **(3) weight space generation**, which explores how new parameters can be produced via generative models, hypernetworks, or model merging. This framing highlights weight space learning as distinct from function-space or purely optimization-centric views, aiming to build a systematic foundation for reasoning about, operating on, and reusing neural network parameters.

## Table of Contents

- [Awesome-Weight-Space-Learning ](#awesome-weight-space-learning-)
  - [Weight Space Learning](#weight-space-learning)
  - [Table of Contents](#table-of-contents)
  - [Weight Space Understanding](#weight-space-understanding)
    - [Structural Foundations](#structural-foundations)
      - [Invariance](#invariance)
      - [Equivariance](#equivariance)
    - [Practical Implications](#practical-implications)
      - [Model Compression](#model-compression)
      - [Model Optimization](#model-optimization)
      - [Weight Space Augmentation](#weight-space-augmentation)
  - [Weight Space Representation](#weight-space-representation)
    - [Representation Approaches](#representation-approaches)
      - [Model-based](#model-based)
      - [Model-free](#model-free)
    - [Practical Implications](#practical-implications-1)
      - [Function Prediction](#function-prediction)
      - [Model Retrieval](#model-retrieval)
      - [Model Editing](#model-editing)
  - [Weight Space Generation](#weight-space-generation)
    - [Generation Approaches](#generation-approaches)
      - [Hypernetworks](#hypernetworks)
      - [Generative Models](#generative-models)
    - [Practical Implications](#practical-implications-2)
      - [Conditional Weight Generation](#conditional-weight-generation)
      - [Real-Time Weight Optimization](#real-time-weight-optimization)
      - [Model Merging](#model-merging)
      - [Weight Initialization](#weight-initialization)
      - [Training Acceleration](#training-acceleration)
      - [Data Generation](#data-generation)
  - [Applications to Related Domains](#applications-to-related-domains)
    - [Implicit Neural Representations](#implicit-neural-representations)
    - [Model Unification](#model-unification)
    - [Continual Leanring](#continual-leanring)
    - [Meta Learning](#meta-learning)
    - [Federated Learning](#federated-learning)
    - [Neural Architecture Search](#neural-architecture-search)
  - [Benchmarks](#benchmarks)
    - [Model Zoo](#model-zoo)
  - [Others](#others)
    - [Survey](#survey)
    - [Thesis](#thesis)
  - [Citation](#citation)

## Weight Space Understanding


### Structural Foundations


#### Invariance


- **[ICML 24]** Improved generalization of weight space networks via augmentations [[PDF](https://arxiv.org/abs/2402.04081)] [[Code](https://github.com/AvivSham/deep-weight-space-augmentations)]
- **[NeurIPS-NeurReps 23]** Data Augmentations in Deep Weight Spaces [[PDF](https://arxiv.org/abs/2311.08851)]
- **[2021]** Lossless Compression of Structured Convolutional Models via Lifting [[PDF](https://arxiv.org/abs/2007.06567)] [[Code](https://github.com/GustikS/NeuraLifting)]
- **[ICLR 23]** Git Re-Basin: Merging Models modulo Permutation Symmetries [[PDF](https://arxiv.org/abs/2209.04836)] [[Code](https://github.com/samuela/git-re-basin)]
- **[ICLR 22]** The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks [[PDF](https://arxiv.org/abs/2110.06296)] [[Code](https://github.com/Neurips21Permutation/PermutationInvariance)]
- **[2022]** Weight-space symmetry in deep networks gives rise to permutation saddles, connected by equal-loss valleys across the loss landscape [[PDF](https://arxiv.org/abs/1907.02911)]
- **[ICML 21]** Geometry of the loss landscape in overparameterized neural networks: Symmetries and invariances [[PDF](https://arxiv.org/abs/2105.12221)]
- **[2025]** Understanding Mode Connectivity via Parameter Space Symmetry [[PDF](https://arxiv.org/abs/2505.23681)]

#### Equivariance


- **[2021]** Universal approximation and model compression for radial neural networks [[PDF](https://arxiv.org/abs/2107.02550)]
- **[2025]** Generalized Linear Mode Connectivity for Transformers [[PDF](https://arxiv.org/abs/2506.22712)]
- **[ICML 23]** Equivariant Architectures for Learning in Deep Weight Spaces [[PDF](https://arxiv.org/abs/2301.12780)] [[Code](https://github.com/AvivNavon/DWSNets)]
- **[NeurIPS 23]** Permutation Equivariant Neural Functionals [[PDF](https://arxiv.org/abs/2302.14040)] [[Code](https://github.com/AllanYangZhou/nfn)]
- **[NeurIPS 24]** Universal neural functionals [[PDF](https://arxiv.org/abs/2402.05232)] [[Code](https://github.com/AllanYangZhou/universal_neural_functional)]
- **[CVPR 25]** Few-shot Implicit Function Generation via Equivariance [[PDF](https://arxiv.org/abs/2501.01601)] [[Code](https://github.com/JeanDiable/EquiGen)]

### Practical Implications


#### Model Compression


- **[2021]** Lossless Compression of Structured Convolutional Models via Lifting [[PDF](https://arxiv.org/abs/2007.06567)] [[Code](https://github.com/GustikS/NeuraLifting)]
- **[2021]** Universal approximation and model compression for radial neural networks [[PDF](https://arxiv.org/abs/2107.02550)]
- **[CVPR 21]** Permute, quantize, and fine-tune: Efficient compression of neural networks [[PDF](https://arxiv.org/abs/2010.15703)]
- **[MIPR 24]** TQCompressor: improving tensor decomposition methods in neural networks via permutations [[PDF](https://arxiv.org/abs/2401.16367)]
- **[ICLR 24]** Merge, Then Compress: Demystify Efficient {SM}oE with Hints from Its Routing Policy [[PDF](https://arxiv.org/abs/2310.01334)]

#### Model Optimization


- **[TMLR 23]** Weight-balancing fixes and flows for deep learning [[PDF](https://openreview.net/pdf?id=uaHyXxyp2r)]
- **[NeurIPS 15]** Path-sgd: Path-normalized optimization in deep neural networks [[PDF](https://arxiv.org/abs/1506.02617)] [[Code](https://github.com/bneyshabur/path-sgd)]
- **[ICLR 19]** G-SGD: Optimizing ReLU Neural Networks in its Positively Scale-Invariant Space [[PDF](https://arxiv.org/abs/1802.03713)]
- **[PR 20]** Projection based weight normalization: Efficient method for optimization on oblique manifold in DNNs [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320320301175)]
- **[UAI 22]** Accelerating training of batch normalization: A manifold perspective [[PDF](https://arxiv.org/abs/2101.02916)]
- **[Mathematics 23]** Neural optimizer adaptations for weight spaces [[PDF](https://arxiv.org/abs/2012.01118)]
- **[NeurIPS 20]** Optimizing deep models: practical methods [[PDF](https://arxiv.org/abs/2009.02439)] [[Code](https://github.com/IBM/NeuronAlignment)]

#### Weight Space Augmentation


- **[NeurIPS-NeurReps 23]** Data Augmentations in Deep Weight Spaces [[PDF](https://arxiv.org/abs/2311.08851)]
- **[ICML 24]** Improved generalization of weight space networks via augmentations [[PDF](https://arxiv.org/abs/2402.04081)] [[Code](https://github.com/AvivSham/deep-weight-space-augmentations)]
- **[CVPR 25]** Few-shot Implicit Function Generation via Equivariance [[PDF](https://arxiv.org/abs/2501.01601)] [[Code](https://github.com/JeanDiable/EquiGen)]
- **[ICML 24]** Equivariant Deep Weight Space Alignment [[PDF](https://arxiv.org/abs/2310.13397)] [[Code](https://github.com/AvivNavon/deep-align)]


## Weight Space Representation


### Representation Approaches


#### Model-based


- **[ECAL 20]** Classifying the classifier: dissecting the weight space of neural networks [[PDF](https://arxiv.org/abs/2002.05688)] [[Code](https://github.com/gabrieleilertsen/nws)]
- Predicting Neural Network Accuracy from Weights [[PDF](https://arxiv.org/abs/2002.11448)] [[Code](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy)]
- **[Natural Communications 21]** Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data [[PDF](https://arxiv.org/abs/2002.06716)] [[Code](https://github.com/CalculatedContent/ww-trends-2021)]
- **[ICML 23]** Equivariant Architectures for Learning in Deep Weight Spaces [[PDF](https://arxiv.org/abs/2301.12780)] [[Code](https://github.com/AvivNavon/DWSNets)]
- **[NeurIPS 23]** Permutation Equivariant Neural Functionals [[PDF](https://arxiv.org/abs/2302.14040)] [[Code](https://github.com/AllanYangZhou/nfn)]
- **[NeurIPS 24]** Universal neural functionals [[PDF](https://arxiv.org/abs/2402.05232)] [[Code](https://github.com/AllanYangZhou/universal_neural_functional)]
- **[NeurIPS 17]** Deep sets [[PDF](https://arxiv.org/abs/1703.06114)] [[Code](https://github.com/manzilzaheer/DeepSets)]
- **[ICML 18]** Deep models of interactions across sets [[PDF](https://arxiv.org/abs/1803.02879)]
- **[NeurIPS 23]** Permutation Equivariant Neural Functionals [[PDF](https://arxiv.org/abs/2302.14040)] [[Code](https://github.com/AllanYangZhou/nfn)]
- **[NeurIPS 21]** Self-supervised representation learning on neural network weights for model characteristic prediction [[PDF](https://arxiv.org/abs/2110.15288)] [[Code](https://github.com/HSG-AIML/NeurIPS_2021-Weight_Space_Learning)]
- **[NeurIPS 23]** Neural Functional Transformers [[PDF](https://arxiv.org/abs/2305.13546)] [[Code](https://github.com/AllanYangZhou/nfn)]
- **[ICML 25]** Equivariant Polynomial Functional Networks [[PDF](https://arxiv.org/abs/2410.04213)] [[Code](https://github.com/Fsoft-AIC/MAGEP-NFN)]
- **[NeurIPS 24]** Monomial matrix group equivariant neural functional networks [[PDF](https://arxiv.org/abs/2409.11697)] [[Code](https://github.com/MathematicalAI-NUS/Monomial-NFN)]
- **[ICML 25]** Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion [[PDF](https://arxiv.org/abs/2502.00264)] [[Code](https://github.com/zhengzaiyi/RotationSymmetry)]
- **[ICLR 25]** Revisiting Multi-Permutation Equivariance through the Lens of Irreducible Representations [[PDF](https://arxiv.org/abs/2410.06665)] [[Code](https://github.com/yonatansverdlov/SchurNet)]
- **[ICLR 23]** NeRN: Learning Neural Representations for Neural Networks [[PDF](https://openreview.net/forum?id=9gfir3fSy3J)] [[Code](https://github.com/maorash/NeRN?utm_source=catalyzex.com)]
- **[NeurIPS 24]** Set-based Neural Network Encoding Without Weight Tying [[PDF](https://arxiv.org/abs/2305.16625)]
- **[ICML-TAGML 23]** On genuine invariance learning without weight-tying [[PDF](https://arxiv.org/abs/2308.03904)]
- **[ICLR 24]** Graph Neural Networks for Learning Equivariant Representations of Neural Networks [[PDF](https://arxiv.org/abs/2403.12143)] [[Code](https://github.com/mkofinas/neural-graphs)]
- **[ICLR 24]** Graph Metanetworks for Processing Diverse Neural Architectures [[PDF](https://arxiv.org/abs/2312.04501)]
- **[NeurIPS 24]** Scale equivariant graph metanetworks [[PDF](https://arxiv.org/abs/2406.10685)] [[Code](https://github.com/jkalogero/scalegmn)]
- **[2025]** Weight Space Representation Learning on Diverse NeRF Architectures [[PDF](https://arxiv.org/abs/2502.09623)]
- **[CVPR 2026]** Weight Space Representation Learning via Neural Field Adaptation [[PDF](https://arxiv.org/abs/2512.01759)] [[Code](https://github.com/inrainbws/wsr.pytorch)]

#### Model-free


- **[2025]** Can this Model Also Recognize Dogs? Zero-Shot Model Search from Weights [[PDF](https://arxiv.org/abs/2502.09619)]
- **[2024]** Deep Linear Probe Generators for Weight Space Learning [[PDF](https://arxiv.org/abs/2410.10811)]
- **[ICLR 24]** Graph Neural Networks for Learning Equivariant Representations of Neural Networks [[PDF](https://arxiv.org/abs/2403.12143)] [[Code](https://github.com/mkofinas/neural-graphs)]
- **[ICML 24]** Learning Useful Representations of Recurrent Neural Network Weight Matrices [[PDF](https://arxiv.org/abs/2403.11998)]
- **[CVPR 25]** Learning on Model Weights using Tree Experts [[PDF](https://arxiv.org/abs/2410.13569)] [[Code](https://horwitz.ai/probex/)]

### Practical Implications


#### Function Prediction


- **[ECAL 20]** Classifying the Classifier: Dissecting the Weight Space of Neural Networks [[PDF](https://arxiv.org/abs/2002.05688)] [[Code](https://github.com/gabrieleilertsen/nws)]
- **[2020]** Predicting Neural Network Accuracy from Weights [[PDF](https://arxiv.org/abs/2002.11448)] [[Code](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy)]
- **[ICLR 24]** Graph Metanetworks for Processing Diverse Neural Architectures [[PDF](https://arxiv.org/abs/2312.04501)]
- **[ICLR 24]** Graph Neural Networks for Learning Equivariant Representations of Neural Networks [[PDF](https://arxiv.org/abs/2403.12143)] [[Code](https://github.com/mkofinas/neural-graphs)]
- **[2024]** Deep Linear Probe Generators for Weight Space Learning [[PDF](https://arxiv.org/abs/2410.10811)]
- **[NeurIPS 22]** Model Zoos: A Dataset of Diverse Populations of Neural Network Models [[PDF](https://arxiv.org/abs/2209.14764)] [[Code](https://github.com/ModelZoos/ModelZooDataset)]
- **[ICLR-SNN 23]** Sparsified Model Zoo Twins: Investigating Populations of Sparsified Neural Network Models [[PDF](https://arxiv.org/abs/2304.13718)] [[Code](https://github.com/ModelZoos/ModelZooDataset)]
- **[ICML 24]** Towards Scalable and Versatile Weight Space Learning [[PDF](https://arxiv.org/abs/2406.09997)] [[Code](https://github.com/HSG-AIML/SANE)]
- **[2025]** Learning Model Representations Using Publicly Available Model Hubs [[PDF](https://arxiv.org/abs/2510.02096)]

#### Model Retrieval


- **[2025]** Can this Model Also Recognize Dogs? Zero-Shot Model Search from Weights [[PDF](https://arxiv.org/abs/2502.09619)]
- **[CVPR 25]** Learning on Model Weights using Tree Experts [[PDF](https://arxiv.org/abs/2410.13569)] [[Code](https://horwitz.ai/probex/)]
- **[2025]** We Should Chart an Atlas of All the World's Models [[PDF](https://arxiv.org/abs/2503.10633)] [[Code](https://horwitz.ai/model-atlas)]

#### Model Editing


- **[NeurIPS 24]** Interpreting the Weight Space of Customized Diffusion Models [[PDF](https://arxiv.org/abs/2406.09413)] [[Code](https://snap-research.github.io/weights2weights/)]
- **[ICML 24]** Towards Scalable and Versatile Weight Space Learning [[PDF](https://arxiv.org/abs/2406.09997)] [[Code](https://github.com/HSG-AIML/SANE)]
- **[NeurIPS 23]** Permutation Equivariant Neural Functionals [[PDF](https://arxiv.org/abs/2302.14040)] [[Code](https://github.com/AllanYangZhou/nfn)]
- **[ICLR 24]** Graph Metanetworks for Processing Diverse Neural Architectures [[PDF](https://arxiv.org/abs/2312.04501)]
- **[ICLR 24]** Graph Neural Networks for Learning Equivariant Representations of Neural Networks [[PDF](https://arxiv.org/abs/2403.12143)] [[Code](https://github.com/mkofinas/neural-graphs)]
- **[NeurIPS 23]** Neural Functional Transformers [[PDF](https://arxiv.org/abs/2305.13546)] [[Code](https://github.com/AllanYangZhou/nfn)]


## Weight Space Generation


### Generation Approaches


#### Hypernetworks


- **[ICLR 17]** HyperNetworks  [[PDF](https://arxiv.org/abs/1609.09106)] [[Code](https://github.com/g1910/HyperNetworks)]
- **[2017]** Bayesian Hypernetworks [[PDF](https://arxiv.org/abs/1710.04759)]
- **[2017]** Implicit weight uncertainty in neural networks [[PDF](https://arxiv.org/abs/1711.01297)] [[Code](https://github.com/pawni/BayesByHypernet)]
- **[ICLR 18]** SMASH: One-Shot Model Architecture Search through HyperNetworks [[PDF](https://arxiv.org/abs/1708.05344)] [[Code](https://github.com/Lornatang/SMASH-PyTorch)]
- **[ICLR 19]** Graph HyperNetworks for Neural Architecture Search [[PDF](https://arxiv.org/abs/1810.05749)]
- **[NeurIPS 21]** Parameter prediction for unseen deep architectures [[PDF](https://arxiv.org/abs/2110.13100)] [[Code](https://github.com/facebookresearch/ppuda)]
- **[ICLR 20]** Continual learning with hypernetworks [[PDF](https://arxiv.org/abs/1906.00695)] [[Code](https://github.com/chrhenning/hypercl)]
- **[ECCV 20]** DHP: Differentiable Meta Pruning via HyperNetworks [[PDF](https://arxiv.org/abs/2003.13683)] [[Code](https://github.com/ofsoundof/dhp)]
- **[CVPR 22]** Sylph: A Hypernetwork Framework  for Incremental Few-shot Object Detection [[PDF](https://arxiv.org/abs/2203.13903)]
- **[TIP 24]** Learning to Generate Parameters of ConvNets for Unseen Image Data [[PDF](https://arxiv.org/abs/2310.11862)] [[Code](https://github.com/tulerfeng/PudNet)]
- **[ICCV 19]** Deep meta functionals for shape representation [[PDF](https://arxiv.org/abs/1908.06277)]
- **[ICML 21]** Personalized Federated Learning using Hypernetworks [[PDF](https://arxiv.org/abs/2103.04628)] [[Code](https://github.com/AvivSham/pFedHN)]
- **[NeurIPS-ML 21]** Meta-learning via hypernetworks [[PDF](https://meta-learn.github.io/2020/papers/38_paper.pdf)]
- **[CVPR 21]** Hyperseg: Patch-wise hypernetwork for real-time semantic segmentation [[PDF](https://arxiv.org/abs/2012.11582)]
- **[ACL-IJCNLP 21]** Parameterefficient multi-task fine-tuning for transformers via shared hypernetworks [[PDF](https://arxiv.org/pdf/2106.04489)] [[Code](https://github.com/rabeehk/hyperformer)]
- **[CVPR 24]** Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models [[PDF](https://arxiv.org/abs/2307.06949)] [[Code](https://github.com/JiauZhang/hyperdreambooth)]
- **[CVPR 22]** HyperStyle: StyleGAN Inversion With HyperNetworks for Real Image Editing [[PDF](https://arxiv.org/abs/2111.15666)] [[Code](https://github.com/yuval-alaluf/hyperstyle)]
- **[AAAI 24]** Hypereditor: Achieving both authenticity and cross-domain capability in image editing via hypernetworks [[PDF](https://arxiv.org/abs/2312.13537)]
- **[CVPR 22]** HyperInverter: Improving StyleGAN Inversion via Hypernetwork [[PDF](https://arxiv.org/abs/2112.00719)] [[Code](https://github.com/VinAIResearch/HyperInverter)]
- **[NeurIPS 22]** Polyhistor: Parameter-Efficient Multi-Task Adaptation for Dense Vision Tasks [[PDF](https://arxiv.org/abs/2210.03265)]
- **[ICML 22]** HyperTransformer: Model Generation for Supervised and Semi-Supervised Few-Shot Learning [[PDF](https://arxiv.org/abs/2201.04182)] [[Code](https://github.com/google-research/google-research/tree/master/hypertransformer)]
- **[2018]** Approximating the predictive distribution via adversarially-trained hypernetworks [[PDF](https://chrhenning.github.io/assets/pdf/BDL18Poster.pdf)]
- **[ICML 19]** Hypergan: A generative model for diverse, performant neural networks [[PDF](https://arxiv.org/abs/1901.11058)] [[Code](https://github.com/simeetnayan81/hypergan-pytorch)]

#### Generative Models

- **[ICLR 26]** Weight-Space Linear Recurrent Neural Networks [[PDF](https://arxiv.org/abs/2506.01153)] [[Code](https://github.com/ddrous/warp)]
- **[NeurIPS 21]** Self-supervised representation learning on neural network weights for model characteristic prediction [[PDF](https://arxiv.org/abs/2110.15288)] [[Code](https://github.com/HSG-AIML/NeurIPS_2021-Weight_Space_Learning)]
- **[NeurIPS 22]** Hyper-Representations as Generative Models: Sampling Unseen Neural Network Weights [[PDF](https://arxiv.org/abs/2209.14733)] [[Code](https://github.com/HSG-AIML/NeurIPS_2022-Generative_Hyper_Representations)]
- **[ICLR-WSL 25]** Instruction-Guided Autoregressive Neural Network Parameter Generation [[PDF](https://arxiv.org/abs/2504.02012)]
- **[ICLR-WSL 25]** Structure Is Not Enough: Leveraging Behavior for Neural Network Weight Reconstruction [[PDF](https://arxiv.org/abs/2503.17138)] [[Code](https://github.com/HSG-AIML/ICLR_WSL_2025-Structure_is_not_enough)]
- Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights [[PDF](https://arxiv.org/abs/2506.16406)] [[Code](https://jerryliang24.github.io/DnD/)]
- **[IJCAI 25]** In-Context Meta LoRA Generation [[Code](https://github.com/YihuaJerry/ICM-LoRA)]
- **[ICML 24]** Towards Scalable and Versatile Weight Space Learning [[PDF](https://arxiv.org/abs/2406.09997)] [[Code](https://github.com/HSG-AIML/SANE)]
- **[2025]** Learning Model Representations Using Publicly Available Model Hubs [[PDF](https://arxiv.org/abs/2510.02096)]
- **[ICLR-WSL 25]** Flow to Learn: Flow Matching on Neural Network Parameters [[PDF](https://arxiv.org/abs/2503.19371)]
- **[2025]** NeuroGen: Neural Network Parameter Generation via Large Language Models [[PDF](https://arxiv.org/abs/2505.12470)]
- **[2022]** Learning to Learn with Generative Models of Neural Network Checkpoints [[PDF](https://arxiv.org/abs/2209.12892)] [[Code](https://www.wpeebles.com/Gpt)]
- **[ICCV 23]** HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion [[PDF](https://arxiv.org/abs/2303.17015)] [[Code](https://ziyaerkoc.com/hyperdiffusion/)]
- **[ICLR 24]** Spatio-Temporal Few-Shot Learning via Diffusive Neural Network Generation [[PDF](https://arxiv.org/abs/2402.11922)] [[Code](https://github.com/tsinghua-fib-lab/GPD)]
- **[2024]** BEND: Bagging Deep Learning Training Based on Efficient Neural Network Diffusion [[PDF](https://arxiv.org/abs/2403.15766)]
- **[2024]** Neural Network Diffusion [[PDF](https://arxiv.org/abs/2402.13144)] [[Code](https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion)]
- **[MM 25]** Text2Weight: Bridging Natural Language and Neural Network Weight Spaces [[PDF](https://arxiv.org/abs/2508.13633)]
- **[2024]** DiffLoRA: Generating Personalized Low-Rank Adaptation Weights with Diffusion [[PDF](https://arxiv.org/abs/2408.06740)]
- **[2025]** Recurrent Diffusion for Large-Scale Parameter Generation [[PDF](https://arxiv.org/abs/2501.11587)] [[Code](https://github.com/NUS-HPC-AI-Lab/Recurrent-Parameter-Generation)]
- **[2025]** ORAL: Prompting Your Large-Scale LoRAs via Conditional Recurrent Diffusion [[PDF](https://arxiv.org/abs/2503.24354)]
- **[ICLR 25]** Diffusion-Based Neural Network Weights Generation [[PDF](https://arxiv.org/abs/2402.18153)] [[Code](https://openreview.net/forum?id=j8WHjM9aMm)]
- **[ICLR 26]** LoRAGen: Structure-Aware Weight Space Learning for LoRA Generation [[PDF](https://openreview.net/pdf?id=mrafO7aTYj)]


### Practical Implications


#### Conditional Weight Generation


- **[CVPR 22]** Sylph: A Hypernetwork Framework  for Incremental Few-shot Object Detection [[PDF](https://arxiv.org/abs/2203.13903)]
- **[TMLR 23]** Meta-Learning via Classifier(-free) Diffusion Guidance [[PDF](https://arxiv.org/abs/2210.08942?utm_source=chatgpt.com)]
- **[MM 25]** Text2Weight: Bridging Natural Language and Neural Network Weight Spaces [[PDF](https://arxiv.org/abs/2508.13633)]
- **[2025]** Continual Adaptation: Environment-Conditional Parameter Generation for Object Detection in Dynamic Scenarios [[PDF](https://arxiv.org/abs/2506.24063)]

#### Real-Time Weight Optimization


- **[CVPR 21]** Hyperseg: Patch-wise hypernetwork for real-time semantic segmentation [[PDF](https://arxiv.org/abs/2012.11582)]
- **[CVPR 22]** HyperStyle: StyleGAN Inversion With HyperNetworks for Real Image Editing [[PDF](https://arxiv.org/abs/2111.15666)] [[Code](https://github.com/yuval-alaluf/hyperstyle)]
- **[ICML-EXAIT 25]** Reimagining Parameter Space Exploration with Diffusion Models [[PDF](https://arxiv.org/abs/2506.17807)]

#### Model Merging


- **[NeurIPS 22]** Hyper-Representations as Generative Models: Sampling Unseen Neural Network Weights [[PDF](https://arxiv.org/abs/2209.14733)] [[Code](https://github.com/HSG-AIML/NeurIPS_2022-Generative_Hyper_Representations)]
- **[2025]** Generative Modeling of Weights: Generalization or Memorization? [[PDF](https://arxiv.org/abs/2506.07998)] [[Code](https://github.com/boyazeng/weight_memorization)]
- **[ICML 24]** Equivariant Deep Weight Space Alignment [[PDF](https://arxiv.org/abs/2310.13397)] [[Code](https://github.com/AvivNavon/deep-align)]
- **[ICML 23]** Equivariant Architectures for Learning in Deep Weight Spaces [[PDF](https://arxiv.org/abs/2301.12780)] [[Code](https://github.com/AvivNavon/DWSNets)]

#### Weight Initialization


- **[CVPR 24]** Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models [[PDF](https://arxiv.org/abs/2307.06949)] [[Code](https://github.com/JiauZhang/hyperdreambooth)]
- **[2022]** Learning to Learn with Generative Models of Neural Network Checkpoints [[PDF](https://arxiv.org/abs/2209.12892)] [[Code](https://www.wpeebles.com/Gpt)]
- **[ICML 23]** Can We Scale Transformers to Predict Parameters of Diverse ImageNet Models? [[PDF](https://arxiv.org/abs/2303.04143)] [[Code](https://github.com/SamsungSAILMontreal/ghn3)]
- **[ICLR 25]** Accelerating Training with Neuron Interaction and Nowcasting Networks [[PDF](https://arxiv.org/abs/2409.04434)] [[Code](https://github.com/SamsungSAILMontreal/nino)]

#### Training Acceleration


- **[ICML 23]** Learning to Boost Training by Periodic Nowcasting Near Future Weights [[PDF](https://openreview.net/forum?id=zHDdkb8LRQ)] [[Code](https://github.com/jjh6297/WNN)]
- **[ICLR 25]** Accelerating Training with Neuron Interaction and Nowcasting Networks [[PDF](https://arxiv.org/abs/2409.04434)] [[Code](https://github.com/SamsungSAILMontreal/nino)]

#### Data Generation


- **[AISTATS 22]** Generative Models as Distributions of Functions [[PDF](https://proceedings.mlr.press/v151/dupont22a.html)] [[Code](https://github.com/EmilienDupont/neural-function-distributions)]
- **[ECCV 24]** Neural Metamorphosis [[PDF](https://arxiv.org/abs/2410.11878)] [[Code](https://adamdad.github.io/neumeta/)]
- **[CVPR 21]** pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis [[PDF](https://arxiv.org/abs/2012.00926)] [[Code](https://github.com/ssumin6/pigan)] 




## Applications to Related Domains

### Implicit Neural Representations
- **[NeurIPS-NeurReps 23]** Data Augmentations in Deep Weight Spaces [[PDF](https://arxiv.org/abs/2311.08851)]
- **[ICML 24]** Improved Generalization of Weight Space Networks via Augmentations [[PDF](https://arxiv.org/abs/2402.04081)] [[Code](https://github.com/AvivSham/deep-weight-space-augmentations)]
- **[CVPR 25]** Few-shot Implicit Function Generation via Equivariance [[PDF](https://arxiv.org/abs/2501.01601)] [[Code](https://github.com/JeanDiable/EquiGen)]
- **[NeurIPS 23]** Neural Functional Transformers [[PDF](https://arxiv.org/abs/2305.13546)] [[Code](https://github.com/AllanYangZhou/nfn)]
- **[ICLR 23]** Deep Learning on Implicit Neural Representations of Shapes [[PDF](https://arxiv.org/abs/2302.05438)] [[Code](https://github.com/CVLAB-Unibo/inr2vec)]
- **[ICML 22]** From data to functa: Your data point is a function and you can treat it like one [[PDF](https://arxiv.org/abs/2201.12204)] [[Code](https://github.com/JurrivhLeon/Functa_pytorch_version)]
- Spatial Functa: Scaling Functa to ImageNet Classification and Generation [[PDF](https://arxiv.org/abs/2302.03130)] [[Code](https://github.com/samuelepapa/spatial_functa)]
- From MLP to NeoMLP: Leveraging Self-Attention for Neural Fields [[PDF](https://arxiv.org/abs/2412.08731)] [[Code](https://github.com/mkofinas/neomlp)]
- **[NeurIPS 20]** Graf: Generative radiance fields for 3d-aware image synthesis [[PDF](https://arxiv.org/abs/2007.02442)] [[Code](https://github.com/autonomousvision/graf)]
- **[CVPR 21]** pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis [[PDF](https://arxiv.org/abs/2012.00926)] [[Code](https://github.com/ssumin6/pigan)]
- **[ICCV 23]** HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion [[PDF](https://arxiv.org/abs/2303.17015)] [[Code](https://ziyaerkoc.com/hyperdiffusion/)]
- **[ECCV 24]** Neural Metamorphosis [[PDF](https://arxiv.org/abs/2410.11878)] [[Code](https://adamdad.github.io/neumeta/)]
- **[CVPR 25]** End-to-End Implicit Neural Representations for Classification [[PDF](https://arxiv.org/abs/2503.18123)] [[Code](https://github.com/SanderGielisse/MWT)]

### Model Unification
- **[NeurIPS 21]** Learning signal-agnostic manifolds of neural fields [[PDF](https://arxiv.org/abs/2111.06387)] [[Code](https://github.com/yilundu/gem)]
- **[ICML 22]** From data to functa: Your data point is a function and you can treat it like one [[PDF](https://arxiv.org/abs/2201.12204)] [[Code](https://github.com/JurrivhLeon/Functa_pytorch_version)]
- Spatial Functa: Scaling Functa to ImageNet Classification and Generation [[PDF](https://arxiv.org/abs/2302.03130)] [[Code](https://github.com/samuelepapa/spatial_functa)]
- GNN-based Unified Deep Learning [[PDF](https://arxiv.org/abs/2508.10583)] [[Code](https://github.com/basiralab/uGNN)]

### Continual Leanring
- **[ICLR 20]** Continual learning with hypernetworks [[PDF](https://arxiv.org/abs/1906.00695)] [[Code](https://github.com/chrhenning/hypercl)]
- **[NeurIPS 24]** Weight Diffusion for Future: Learn to Generalize in Non-Stationary Environments [[PDF](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0c1124bd3be769dacf491d92d499c7d8-Abstract-Conference.html)] [[Code](https://github.com/BIT-DA/W-Diff)]
- **[2025]** Continual Adaptation: Environment-Conditional Parameter Generation for Object Detection in Dynamic Scenarios [[PDF](https://arxiv.org/abs/2506.24063)]

### Meta Learning
- **[AAAI 24]** MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning [[PDF](https://arxiv.org/abs/2307.16424?utm_source=chatgpt.com)]
- Learning to Learn Weight Generation via Local Consistency Diffusion [[PDF](https://arxiv.org/abs/2502.01117)]
- **[NeurIPS-ML 21]** Meta-learning via hypernetworks [[PDF](https://meta-learn.github.io/2020/papers/38_paper.pdf)]

### Federated Learning
- **[CIKM 24]** Beyond Aggregation: Efficient Federated Model Consolidation with Heterogeneity-Adaptive Weights Diffusion [[PDF](https://dl.acm.org/doi/10.1145/3627673.3679879)]
- **[ICML 21]** Personalized Federated Learning using Hypernetworks [[PDF](https://arxiv.org/abs/2103.04628)] [[Code](https://github.com/AvivSham/pFedHN)]
- **[AAAI 25]** pFedGPA: Diffusion-based Generative Parameter Aggregation for Personalized Federated Learning [[PDF](https://arxiv.org/abs/2409.05701)]

### Neural Architecture Search
- **[ICLR 19]** Graph HyperNetworks for Neural Architecture Search [[PDF](https://arxiv.org/abs/1810.05749)]
- **[NeurIPS 21]** Parameter prediction for unseen deep architectures [[PDF](https://arxiv.org/abs/2110.13100)] [[Code](https://github.com/facebookresearch/ppuda)]


## Benchmarks


### Model Zoo


- **[NeurIPS 22]** Model Zoos: A Dataset of Diverse Populations of Neural Network Models [[PDF](https://arxiv.org/abs/2209.14764)] [[Code](https://github.com/ModelZoos/ModelZooDataset)]
- **[ICLR-SNN 23]** Sparsified Model Zoo Twins: Investigating Populations of Sparsified Neural Network Models [[PDF](https://arxiv.org/abs/2304.13718)] [[Code](https://github.com/ModelZoos/ModelZooDataset)]
- **[ICLR 25]** Unsupervised Model Tree Heritage Recovery [[PDF](https://arxiv.org/abs/2405.18432)] [[Code](https://horwitz.ai/mother)]
- **[NeurIPS 24]** Implicit-Zoo: A Large-Scale Dataset of Neural Implicit Functions for 2D Images and 3D Scenes [[PDF](https://arxiv.org/abs/2406.17438)] [[Code](https://github.com/qimaqi/Implicit-Zoo/)]
- **[NeurIPS 24]** Interpreting the Weight Space of Customized Diffusion Models [[PDF](https://arxiv.org/abs/2406.09413)] [[Code](https://snap-research.github.io/weights2weights/)]
<!-- - Model Zoos for Benchmarking Phase Transitions in Neural Networks [[PDF](https://openreview.net/forum?id=JlkqReTftJ)] -->
<!-- - We Should Chart an Atlas of All the World's Models [[PDF](https://arxiv.org/abs/2503.10633)] [[Code](https://horwitz.ai/model-atlas)] -->
- Scaling LLaNA: Advancing NeRF-Language Understanding Through Large-Scale Training [[PDF](https://arxiv.org/abs/2504.13995)] [[Code](https://andreamaduzzi.github.io/llana/)]
- **[ICLR-WSL 25]** A Model Zoo of Vision Transformers [[PDF](https://arxiv.org/abs/2504.10231)] [[Code](http://github.com/ModelZoos/ViTModelZoo)]
- **[Electronics]** An Open Dataset of Neural Networks for Hypernetwork Research [[PDF](https://www.mdpi.com/2079-9292/14/14/2831)]
- **[ICCS 25]** Towards Weight-Space Interpretation of Low-Rank Adapters for Diffusion Models [[PDF](https://www.iccs-meeting.org/archive/iccs2025/papers/159030108.pdf)]
- **[ECAL 20]** Classifying the classifier: dissecting the weight space of neural networks [[PDF](https://arxiv.org/abs/2002.05688)] [[Code](https://github.com/gabrieleilertsen/nws)]
- Predicting Neural Network Accuracy from Weights [[PDF](https://arxiv.org/abs/2002.11448)] [[Code](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy)]
- **[NeurIPS 21]** Self-supervised representation learning on neural network weights for model characteristic prediction [[PDF](https://arxiv.org/abs/2110.15288)] [[Code](https://github.com/HSG-AIML/NeurIPS_2021-Weight_Space_Learning)]
- **[ICML 24]** Learning Useful Representations of Recurrent Neural Network Weight Matrices [[PDF](https://arxiv.org/abs/2403.11998)]


## Others


### Survey


- A Brief Review of Hypernetworks in Deep Learning [[PDF](https://arxiv.org/abs/2306.06955)]
- Implicit Neural Representation in Medical Imaging: A Comparative Survey [[PDF](https://arxiv.org/abs/2307.16142)]
- Learning from Models Beyond Fine-Tuning [[PDF](https://www.nature.com/articles/s42256-024-00961-0)]
- Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities [[PDF](https://arxiv.org/abs/2408.07666)]
- Where Do We Stand with Implicit Neural Representations? A Technical and Performance Survey [[PDF](https://arxiv.org/abs/2411.03688)]
- Symmetry in Neural Network Parameter Spaces [[PDF](https://arxiv.org/abs/2506.13018)]
- **[EDBT 25]** Model Lakes [[PDF](https://arxiv.org/abs/2403.02327)]


### Thesis


- **[PhD Thesis]** Hyper-Representations: Learning from Populations of Neural Networks [[PDF](https://arxiv.org/abs/2410.05107)]
- **[PhD Thesis]** Acquiring and Adapting Priors for Novel Tasks via Neural Meta-Architectures [[PDF](https://arxiv.org/abs/2507.10446)]
- **[MSc Thesis]** Geometric Flow Models over Neural Network Weights [[PDF](https://arxiv.org/abs/2504.03710)] [[Code](https://github.com/ege-erdogan/weightflow)]


## Citation

> [!NOTE]
> If you find this repository useful for your research, please consider citing our survey and starring this repository.  
> This project is intended to serve as a continuously updated reading list for weight space learning, covering understanding, representation, and generation.  
> Contributions are welcome. If you notice missing papers, broken links, or categorization issues, feel free to open an issue or submit a pull request.


If you would like to cite this repository, you can use:

```bibtex
@article{han2026wsl_survey,
  title   = {A Survey of Weight Space Learning: Understanding, Representation, and Generation},
  author  = {Han, Xiaolong and Wang, Zehong and Zhao, Bo and Zhang, Binchi and Li, Jundong and Borth, Damian and Yu, Rose and Maron, Haggai and Ye, Yanfang and Yin, Lu and Neri, Ferrante},
  journal = {arXiv preprint arXiv:2603.10090},
  year    = {2026}
}
