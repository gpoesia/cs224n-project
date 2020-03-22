# Inside Out Auto-Completion

  Auto-complete has become a ubiquitous feature in modern-day typing systems. Traditional systems typically try to predict the next characters or words the user is about to type. In contrast, keyword-based auto-completion systems expect the user to type keywords,
   then fill in the least informative parts of the sentence, and have been successful
  for a variety of natural language tasks. In this project, we implement a keyword-based auto-complete technique for programming languages,
  as auto-complete systems in modern IDEs are still mostly left-to-right. We use three programming
  languages with different verbosity levels: Python, Haskell and Java. We evaluate different encoding schemes paired
  with a neural decoder with various trade-offs in translation accuracy and compression, and evaluate their accuracy and robustness.

## Getting Started

``` bash
    pip install {numpy, pandas, torch, matplotlib, jupyter}
    jupyter-notebook Explanation.ipynb
```
This first notebook will give a good idea of how to use this notebook.
To see how to train models, see `Experiments.ipynb`, and for other bits of analysis see `Results.ipynb`.
Lastly, to view a paper about this repository, see the the WriteUp directory.

## Authors

**Scott Viteri, Lauren Gillespie, Gabriel Poesia**

## Acknowledgments

* [Mina Lee](https://minalee.info/), our project mentor whose [work](https://arxiv.org/abs/1911.06964)  on keyword-based autocomplete for natural language settings in the natural language setting inspired our project
* Staff of Stanford's class [CS224n](http://web.stanford.edu/class/cs224n/) 
