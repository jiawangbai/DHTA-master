# Targeted Attack for Deep Hashing based Retrieval
This repository provides implementatin our ECCV 2020 work: Targeted Attack for Deep Hashing based Retrieval.

## Abstract

The deep hashing based retrieval method is widely adopted in large-scale image and video retrieval. However, there is little investigation on its security. In this paper, we propose a novel method, dubbed deep hashing targeted attack (DHTA), to study the targeted attack on such retrieval. Specifically, we first formulate the targeted attack as a *point-to-set* optimization, which minimizes the average distance between the hash code of an adversarial example and those of a set of objects with the target label. Then we design a novel *component-voting scheme* to obtain an *anchor code* as the representative of the set of hash codes of objects with the target label, whose optimality guarantee is also theoretically derived. To balance the performance and perceptibility, we propose to minimize the Hamming distance between the hash code of the adversarial example and the anchor code under the <a href="https://www.codecogs.com/eqnedit.php?latex=\ell^\infty" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ell^\infty" title="\ell^\infty" /></a> restriction on the perturbation. Extensive experiments verify that DHTA is effective in attacking both deep hashing based image retrieval and video retrieval. 

## Install
1. Install PyTorch > 1.4
2. Clone this repo and uncompress the pre-trained model inside:
```shell
git clone 
```
3. Download the pretrained hashing model: [VGG11_32_for_IamgeNet](https://drive.google.com/file/d/1V6Nvr0DMhquqWwsl1CQtv0Kug7aXXTzx/view?usp=sharing)
4. Save this model in "models/imagenet_vgg11_32"

## Attack
Run the below command to reproduce our result.

```shell
python attack_imagenet.py --n-anchor 9 --root [imagent-data-root] --reproduce --gpu-id [gpu-id]
```

```shell
python attack_imagenet.py --n-anchor 1 --root [imagent-data-root] --reproduce --gpu-id [gpu-id]
```

## Some Results

The detail README is coming soon...

