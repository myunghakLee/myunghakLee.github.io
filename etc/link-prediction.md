---
layout: page
title: Link Prediction(링크 예측)
description: > 
        Link prediction에 대한 기초적인 설명을 진행하는 포스팅입니다.

---
* toc
{:toc}


## Knowledge Distillation이란

Link prediction은 지식 그래프 완성 작업중 하나이다. 현실 세계의 정보는 너무 방대하며 끊임없이 변하기 때문에지식 그래프는 필연적으로 불완전성을 타고난다. 따라서 지식 그래프를 최대한 완전하게 만들기 위해 우리는 링크 예측과 같은 모델을 이용한다.

![alt text](/images/etc/link-prediction/image.png)

우리는 link prediction 모델을 학습하기 위해 지식 그래프를 triplet dataset(삼중데이터 집합)을 바꾼다(설명의 편이를 위해 [그림 1]의 지식 그래플 조금 축소하였다).

![alt text](/images/etc/link-prediction/image-1.png)

그리고 이렇게 만들어진 삼중 데이터 집합에서 head entity나 tail entity 둘중 하난를 마스킹 처리한 후 나머지 둘을 가지고 마스킹 처리한 것을 맞추는 방식으로 학습한다. 예를 들어 head entity가 마스킹 되어 있다면 relation과 tail을 가지고 이를 맞추는 형식이다.


![alt text](/images/etc/link-prediction/image-2.png){:.centered}

Link prediction 모델은 모델의 구조에 따라 크게 translational based model과 neural network based model로 나뉜다. 지금부터 이에 관해 알아보겠다.

## Translational Based Model

Translational based model은 우선 triplet data로부터 entity set과 relation set을 뽑아온다.

![alt text](/images/etc/link-prediction/image-3.png)

그 후 entity set과 relation set을 모두 벡터로 임베딩 시킨다.

![alt text](/images/etc/link-prediction/image-4.png)

이렇게 만들어진 벡터를 이용해 head entity에 해당하는 벡터 **h**를 relation에 해당하는 벡터 **r**을 이용해 연산을 진행시키면 tail entity에 해당하는 벡터 **t**가 나와야 한다는 수식을 이용해 모델을 학습시킨다. 그리고 이 때 무슨 수식을 사용하느냐에 따라 모델의 종류가 나뉜다. 이번 포스팅에서는 TransE, TransR, HAKE에 대하여 알아보겠다.


### TransE, TransR, HAKE

TransE는 **h**에 **r**을 더하면 **t**가 나온다는 수식을 사용한다. 그리고 TransR은 **h**에 **r**을 곱하면 **t**가 나온다는 수식을 사용한다. 마지막으로 HAKE는 극 좌표계를 사용한 모델로 객체들의 계층을 모델링하기 위해 고안된 모델이다. HAKE의 극좌표계에서 반지름은 객체들의 계층을, 각도는 동일한 계층내에서의 분류를 나타낸다. 이들을 그림으로 그리면 다음과 같다.

![alt text](/images/etc/link-prediction/image-5.png)


## Neural Network Based Model

Neural network based model 역시 [그림 5] 처럼 entity set과 relation set을 벡터로 만드는 부분까지는 동일하다. 단 이렇게 나온 벡터를 neural network에 집어넣어 정답을 구한다는 점이 차이점이다.

![alt text](/images/etc/link-prediction/image-6.png)

