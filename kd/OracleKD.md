---
layout: page
title: Distillation for High-Quality Knowledge Extraction via Explainable Oracle Approach
---

## Knowledge Distillation이란

Knowledge distillation은 대규모 모델의 지식을 작은 모델로 압축하는 기법이다. Knowledge distillation의 목적은 규모가 큰 모델(Teacher model)로 부터 추출된 지식(knowledge)을 더 작은 모델(Student model)로 전달하여, Student model이 Teacher model과 유사한 성능을 발휘하도록 하는것이다.

Knowledge distillation의 과정에서 여러 유형의 지식이 전달될 수 있는데, 이를 response knowledge와 feature knowledge로 구분할 수 있습니다:

![alt text](/images/kd/OracleKD/image-3.png)

**1. Response Knowledge:**

> Response knowledge는 Teacher model의 최종 출력(응답)을 기반으로 한 지식입니다. 즉, Teacher model의 분류 문제에서 각 클래스에 대한 softmax 확률을 Student model에게 제공하여, Student model이 이를 학습하도록 하는것이다. 이러한 확률 값은 단순한 정답 레이블보다 Teacher model의 confidence나 경향을 포함한 더 풍부한 정보를 포함하고 있기 때문에 Student model이 더욱 효율적으로 학습할 수 있다.
이를 통해 Student model이 Teacher model이 예측한 클래스 간의 미묘한 차이를 이해하게 되어, 학습 과정에서 고품질의 지식을 효과적으로 학습할 수 있다.

**2. Feature Knowledge:**
> Feature knowledge는 중간 레이어에서 추출된 피처(특징 맵)을 기반으로 한 지식이다. Teacher model의 중간 레이어에서 학습된 중요한 특징들이 Student model로 전달되어, Student model이 이 특징들을 학습할 수 있게 한다.
> Feature knowledge는 Teacher model이 입력 데이터에서 중요한 패턴을 포착하는 방법을 Student model이 배우는 데 도움을 준다. 이 방법은 모델의 구조적 차이에도 적용 가능하여, Teacher와 Student model의 아키텍처가 다르더라도 유효하게 동작할 수 있다.


## Student Model의 성공적인 학습을 위한 Teacher Model의 역할

이 때 student model을 잘 학습시키기 위해서는 teacher model은 아래와 같은 역할을 준수해야 한다.

* **높은 성능의 모델 유지:** Teacher model은 기본적으로 높은 정확도와 성능을 유지해야 한다. Teacher model의 성능이 낮다면, Student model도 잘못된 패턴을 학습하게 될 가능성이 크다. 이를 위해 Teacher model은 충분한 데이터와 적절한 학습 전략으로 사전 학습이 되어 있어야 하며, 다양한 데이터 분포와 상황에서 좋은 성능을 발휘할 수 있어야 한다.


* **고품질의 Knowledge 제공:** Teacher model은 Student model에 전달할 knowledge의 품질이 높아야 한다. 이 때 knowledge의 품질은 `t-SNE`, `Silhouette Score`, `ECE(Expected Calibration Error)` 등을 통하여 평가 될 수 있다.
    > t-SNE: t-SNE는 고차원 데이터(예: feature knowledge)를 2차원 또는 3차원으로 시각화하여, 데이터가 어떻게 군집을 이루고 있는지 시각적으로 확인하는 데 사용된다. 동일한 class, 혹은 유사한 class간에 군집이 잘 형성되어 있을수록 품질이 높다고 평가할 수 있다.

    > Silhouette Score: Silhouette Score는 clustering 성능을 측정하는 지표로, Teacher model과 Student model이 데이터의 특징을 어떻게 그룹화하는지 평가할 수 있다. 높은 Silhouette Score는 각 군집이 잘 구분되면서도 내부적으로 응집력이 강한 것을 의미하며, 이는 모델이 고품질의 feature knowledge를 학습했음을 나타낸다.

    > ECE: ECE는 모델이 예측한 확률 값과 실제 정답 간의 일치 정도를 나타냅니다. 잘 학습된 Teacher model은 confidence값과 실제 정답을 맞출 확률이 일치할 가능성이 크다. 즉 잘 calibration 된 model은 더 고품질의(즉, 신뢰할 수 있는) 지식을 제공할 수 있다.


## Proposed Method

본 논문에서는 높은 성능과 고품질의 지식을 동시에 제공할 수 있는 새로운 방법을 제시하며, 이를 위해 reinforced data를 활용한다. 이 데이터는 adversarial example과 반대로, 입력 데이터에 쌓인 gradient를 입력 데이터에서 빼는 방식으로 생성된다. 이는 모델의 손실을 최소화하는 방향으로 입력을 수정하는 것이며, 그 결과 아래 표에서 확인할 수 있듯이 모델의 성능이 크게 향상된다.

![alt text](/images/kd/OracleKD/image-5.png)

Reinforced data를 만들고(step A), 이를 이용해 knowledge distillation을 진행하는(step B) 방법은 아래 그림과 같다.

![alt text](/images/kd/OracleKD/image-4.png)

* step A: teacher model의 output과 ground truth로부터 생성된 loss를 이용해 back-propagation 시켜 input data의 loss에 대한 gradient $ df(x)/dx $ 를 구한다. 그리고 input data의 scale을 반영해주기 위해 gradient와 input data를 element wise product한 후 가중치 $ \gamma $를 곱해준다. 그리고 이 값을 input data에 대해 빼준다.

![alt text](/images/kd/OracleKD/image-6.png){: width="200px"}

$$
    \bold{x}

$$


* step B: 그리고 이렇게 만들어진 reinforced data를 다시한번 teacher model의 입력으로 집어 넣어 더 높은 정확도를 지닌 response knowledge를 만들어낸다. 그리고 이를 student model과 kl-divergence를 통해 비교하여 distillation loss를 구한다.
