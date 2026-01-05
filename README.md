# NEURAL2
  
Revamped version of the old archived NEURAL repository.

NEURAL2 is a torch-like collection of neural-network-related classes and structures used for doing simple machine learning tasks. It is modular and supports various different operations. Like VSC (the renderer I made previously), this is made entirely using software.
  
# CREDITS  
  
Modularity is based on PyTorch/Tensorflow. The computational algorithms are based on various resources obtained on the internet, some of which are credited in the runner files.

# PARTICULAR RESOURCES

https://deeplearning.cs.cmu.edu/F21/

# DEDICATIONS  
  
NULL

# HOW TO USE THIS

Only `main.cpp` for now. Forward functions are called `forward` (usually) or `forwards` (usually `operator()` also overrides it), and backwards functions are called `backward` (usually) or `backwards`. Blame the type/inheritance systems and my foolishness in thinking this could just be one little happy polymorphic family. (Maybe I can make `Layer` generic?)

# HOW TO RECREATE THIS  
  
Read through the header files for information on how neural networks learn and operate.

# QUESTIONS/CONCERNS/ISSUES

Feel free to push an issue, do a pull request, or simply contact me (NormalExisting on Discord, Bluesky, and Telegram).

# SHOWCASE!

Confusion matrix for `conv1d` unit test: `[[4886, 1281], [1126, 12707]]`