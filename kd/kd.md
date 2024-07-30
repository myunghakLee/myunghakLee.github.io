---
layout: page
title: Knowledge Distillation(지식 증류)
# description: >
#   Version 9.1 provides minor design changes, new features, and closes multiple issues.

---
* toc
{:toc}

## Knowledge Distillation이란

Knowledge distillation의 가장 대표적인 목적은 모델 경량화입니다. 이를 위해, 크고 복잡한 모델(일명 *teacher model*)의 지식을 보다 작고 효율적인 모델(일명 *student model*)로 전달하는 과정을 거치게 됩니다. 이 때 *student model*은 *teacher model*과 동일한 output을 내도록 학습되며 이 과정을 통해 우리는 규모가 큰 *teacher model*을 규모가 작은 *student model*로 압축하여 경량화 효과를 얻을 수 있습니다. 


## 기본 원리 및 방법론

다시 강조하지만 knowledge distillation의 근본적인 목적은 규모가 작은 *student model*의 output이 *teacher model*의 output과 동일하게 나오도록 하는것입니다. 

그러기 위하여 knowledge distillation은 2가지 distillation 기법을 사용합니다.

하나는 student model과 teacher model의 최종 output을 동일하게 만들기 위한 *response distillation*이고 다른 하나는 intermediate feature 동일하게 만들기 위한 *feature distillation*입니다.


![alt text](/images/kd/kd/image-1.png)

### Response Distillation
Response distillation은 teacher model과 student model의 output의 차이를 kl-divergence를 이용해 구하고 이를 student model을 학습시키기 위한 loss로서 사용합니다. 이를 수식으로 쓰면 다음과 같습니다.

$$\mathcal{L}_{\text{resp}} = D_{KL}(\mathcal{P}_t || \mathcal{P}_s)$$

위 수식에서 $$\mathcal{P}_t = softmax(O_t/\tau)$$로 $$O_t$$는 teacher model의 output이고 $$\tau$$는 softmax의 평활도를 조절하는 temperature parameter입니다. 




### Feature Distillation
Feature Distillation은 teacher model과 student model의 intermediate feature의 차이를 MSE를 이용해 구하고 이를 student model을 학습시키기 위한 loss로서 사용합니다. 이를 수식으로 쓰면 다음과 같습니다.

$$\mathcal{L}_{\text{feat}} = MSE(h_t || h_s)$$

위 수식에서 $$h_t$$는 teacher model의 intermediate layer의 output입니다. 

이 때 몇번째 intermediate layer의 output인 지는 사용자가 정하는 hyper parameter입니다.(해당 hyper parameter를 학습을 통해 보다 세밀하게 다루는 후속 연구가 몇몇 진행되긴 했지만 본 포스팅에서는 다루지 않겠습니다)  

### Distillation Loss

Distillation loss는 앞서 정의한 response loss와 feature loss를 합하여 계산합니다. 

$$\mathcal{L}_{\text{distill}} = \alpha \mathcal{L}_{\text{resp}} + \mathcal{L}_{\text{feat}}$$



### Summary

최종적으로 정리하자면 knowledge distillation은 teacher model과 student model이 동일한 성능을 내도록 만들기 위한 학습을 진행합니다. 그리고 이를 위해 [feature distillation](#feature-distillation)과 [response distillation](#response-distillation)을 이용해 학습을 진행합니다.

물론 이 때 일반적인 cross entropy loss 역시 포함시킵니다. 따라서 student model을 학습시키기 위한 최종 수식은 아래와 같습니다.

$$\mathcal{L}_{\text{feat}} = \beta\mathcal{L}_{\text{distill}} + \mathcal{L}_{\text{CE}}$$
