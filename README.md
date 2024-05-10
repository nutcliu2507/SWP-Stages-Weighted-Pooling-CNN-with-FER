# Abstract:
We propose SWP-net, a lightweight face encoder model that balances parameter and training cycle considerations. The core components of SWP-net comprise four key features: (1) the facial feature extraction layer, (2) the facial feature attention layer, (3) the stages weighted multi-layer perception layer, and (4) the final stages weighted fusion layer. SWP-net offers a novel FER model that prioritizes efficiency while maintaining accuracy. With only 14.16 million parameters and 2.14 GMACs (Giga Multiply accumulate operation per Second), SWP-net achieves competitive accuracy rates on public standard FER datasets, reaching 89.6%, 87.3% and 62.3% on RAF-DB, FERPlus and AffectNet, respectively.  
# SWP: Stages Weights Pooling CNN
![image](https://github.com/nutcliu2507/SWP-Stages-Weighted-Pooling-CNN-with-FER/blob/main/SWP.png)  


# Weight dowload(>25MB):  
RAF-db: [download](https://drive.google.com/file/d/1kKSf6heqtlzXkm0DB4nZWJnUyw0BI9P8/view?usp=sharing)

  
# Environment  
OS: Windows 10  
GPU: Nvidia GeForce GTX 2080 Ti  
Python: 3.8.5  
Troch: 1.7.1+cu110  
Cuda: 11.1.1_456.81  
Cudnn: v8.0.4.300

# Base install linked
Nvidia Driver:  
https://www.nvidia.com/zh-tw/geforce/drivers/  
Python:  
https://www.python.org/downloads/release/python-385/  
Cuda & Cudnn:  
https://developer.nvidia.com/cuda-toolkit-archive  
https://developer.nvidia.com/rdp/cudnn-archive  
