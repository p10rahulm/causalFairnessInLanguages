"""
@misc{tancik2020fourier,
      title={Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains},
      author={Matthew Tancik and Pratul P. Srinivasan and Ben Mildenhall and Sara Fridovich-Keil and Nithin Raghavan and Utkarsh Singhal and Ravi Ramamoorthi and Jonathan T. Barron and Ren Ng},
      year={2020},
      eprint={2006.10739},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@article{long2021rffpytorch,
  title={Random Fourier Features Pytorch},
  author={Joshua M. Long},
  journal={GitHub. Note: https://github.com/jmclong/random-fourier-features-pytorch},
  year={2021}
}
"""


import torch
from rff import layers



if __name__=="__main__":
    X = torch.randn((4, 4, 2))
    print("X = ", X)
    encoding = layers.PositionalEncoding(sigma=1.0, m=10)
    Xmod = encoding(X)
    print("Xmod = ", Xmod)
