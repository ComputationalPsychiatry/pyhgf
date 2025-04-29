# How to cite?

If you are using the *pyhgf* package for your research, we ask you to cite the following paper in the final publication:

> Legrand, N., Weber, L., Waade, P. T., Daugaard, A. H. M., Khodadadi, M., Mikuš, N., & Mathys, C. (2024). pyhgf: A neural network library for predictive coding (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2410.09206

```text
@misc{https://doi.org/10.48550/arxiv.2410.09206,
  doi = {10.48550/ARXIV.2410.09206},
  url = {https://arxiv.org/abs/2410.09206},
  author = {Legrand,  Nicolas and Weber,  Lilian and Waade,  Peter Thestrup and Daugaard,  Anna Hedvig Møller and Khodadadi,  Mojtaba and Mikuš,  Nace and Mathys,  Chris},
  keywords = {Neural and Evolutionary Computing (cs.NE),  Artificial Intelligence (cs.AI),  Machine Learning (cs.LG),  Neurons and Cognition (q-bio.NC),  FOS: Computer and information sciences,  FOS: Computer and information sciences,  FOS: Biological sciences,  FOS: Biological sciences},
  title = {pyhgf: A neural network library for predictive coding},
  publisher = {arXiv},
  year = {2024},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

If your application is using the generalised Hierarchical Gaussian Filer, we also ask you to cite the following publication:

> Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., & Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 2). arXiv. https://doi.org/10.48550/ARXIV.2305.10937 

```text
@misc{https://doi.org/10.48550/arxiv.2305.10937,
  doi = {10.48550/ARXIV.2305.10937},
  url = {https://arxiv.org/abs/2305.10937},
  author = {Weber,  Lilian Aline and Waade,  Peter Thestrup and Legrand,  Nicolas and Møller,  Anna Hedvig and Stephan,  Klaas Enno and Mathys,  Christoph},
  keywords = {Neural and Evolutionary Computing (cs.NE),  Neurons and Cognition (q-bio.NC),  FOS: Computer and information sciences,  FOS: Computer and information sciences,  FOS: Biological sciences,  FOS: Biological sciences},
  title = {The generalized Hierarchical Gaussian Filter},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

If you want to refere to the fundational description of the Hierarchical Gaussian Filter, or other important mathematical derivations, please refer to the following publications:

> Mathys, C. (2011). A Bayesian foundation for individual learning under uncertainty. Frontiers in Human Neuroscience, 5. https://doi.org/10.3389/fnhum.2011.00039  

> Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston, K. J., & Stephan, K. E. (2014). Uncertainty in perception and the hierarchical Gaussian filter. Frontiers in Human Neuroscience, 8. https://doi.org/10.3389/fnhum.2014.00825

```text
@article{2011:mathys,
abstract = {Computational learning models are critical for understanding mechanisms of adaptive behavior. However, the two major current frameworks, reinforcement learning (RL) and Bayesian learning, both have certain limitations. For example, many Bayesian models are agnostic of inter-individual variability and involve complicated integrals, making online learning difficult. Here, we introduce a generic hierarchical Bayesian framework for individual learning under multiple forms of uncertainty (e.g., environmental volatility and perceptual uncertainty). The model assumes Gaussian random walks of states at all but the first level, with the step size determined by the next highest level. The coupling between levels is controlled by parameters that shape the influence of uncertainty on learning in a subject-specific fashion. Using variational Bayes under a mean-field approximation and a novel approximation to the posterior energy function, we derive trial-by-trial update equations which (i) are analytical and extremely efficient, enabling real-time learning, (ii) have a natural interpretation in terms of RL, and (iii) contain parameters representing processes which play a key role in current theories of learning, e.g., precision-weighting of prediction error. These parameters allow for the expression of individual differences in learning and may relate to specific neuromodulatory mechanisms in the brain. Our model is very general: it can deal with both discrete and continuous states and equally accounts for deterministic and probabilistic relations between environmental events and perceptual states (i.e., situations with and without perceptual uncertainty). These properties are illustrated by simulations and analyses of empirical time series. Overall, our framework provides a novel foundation for understanding normal and pathological learning that contextualizes RL within a generic Bayesian scheme and thus connects it to principles of optimality from probability theory.},
author = {Mathys, Christoph D.},
doi = {10.3389/fnhum.2011.00039},
file = {:home/laew/lit/pdf/Mathys - 2011 - A Bayesian foundation for individual learning under uncertainty.pdf:pdf},
isbn = {1662-5161 (Electronic){\$}\backslash{\$}n1662-5161 (Linking)},
issn = {16625161},
journal = {Frontiers in Human Neuroscience},
keywords = {acetylcholine,decision-,dopamine,hierarchical models,neuromodul,neuromodulation,serotonin,variational Bayes,variational bayes},
number = {May},
pages = {1--20},
pmid = {21629826},
title = {{A Bayesian foundation for individual learning under uncertainty}},
url = {http://journal.frontiersin.org/article/10.3389/fnhum.2011.00039/abstract},
volume = {5},
year = {2011}
}

@ARTICLE{2014:mathys,
AUTHOR={Mathys, Christoph D. and Lomakina, Ekaterina I. and Daunizeau, Jean and Iglesias, Sandra and Brodersen, Kay H. and Friston, Karl J. and Stephan, Klaas E.},
TITLE={Uncertainty in perception and the Hierarchical Gaussian Filter},
JOURNAL={Frontiers in Human Neuroscience},
VOLUME={8},
YEAR={2014},
URL={https://www.frontiersin.org/articles/10.3389/fnhum.2014.00825},
DOI={10.3389/fnhum.2014.00825},
ISSN={1662-5161},
ABSTRACT={In its full sense, perception rests on an agent's model of how its sensory input comes about and the inferences it draws based on this model. These inferences are necessarily uncertain. Here, we illustrate how the Hierarchical Gaussian Filter (HGF) offers a principled and generic way to deal with the several forms that uncertainty in perception takes. The HGF is a recent derivation of one-step update equations from Bayesian principles that rests on a hierarchical generative model of the environment and its (in)stability. It is computationally highly efficient, allows for online estimates of hidden states, and has found numerous applications to experimental data from human subjects. In this paper, we generalize previous descriptions of the HGF and its account of perceptual uncertainty. First, we explicitly formulate the extension of the HGF's hierarchy to any number of levels; second, we discuss how various forms of uncertainty are accommodated by the minimization of variational free energy as encoded in the update equations; third, we combine the HGF with decision models and demonstrate the inversion of this combination; finally, we report a simulation study that compared four optimization methods for inverting the HGF/decision model combination at different noise levels. These four methods (Nelder–Mead simplex algorithm, Gaussian process-based global optimization, variational Bayes and Markov chain Monte Carlo sampling) all performed well even under considerable noise, with variational Bayes offering the best combination of efficiency and informativeness of inference. Our results demonstrate that the HGF provides a principled, flexible, and efficient—but at the same time intuitive—framework for the resolution of perceptual uncertainty in behaving agents.}
}
```
