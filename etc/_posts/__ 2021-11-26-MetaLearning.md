---
layout: post
title: Meta Learning
# description: > 
    
---



## Meta Learning이란?

 Meta learinig은 적은 데이터만으로 인공지능 모델을 학습시키는 것이다. 보통 인공지능 모델을 학습시키기 위해서는 많은 양의 데이터가 필요하다. 그러나 사람의 경우를 보면 굉장히 적은 수로도 학습이 가능하다. 예를 들어보면 강아지를 한번도 보지 않은 사람도 굉장히 적은 양의 강아지 사진만으로도 강아지를 분류해 낼 수 있다. 이와 같이 인공지능 모델도 적은 수의 데이터만으로도 학습이 가능하게 만드는 것이  Meta Learning에서 하고싶은 것이다.

Meta Learning의 예시는 다음과 같은 것들이 있다.

* 강아지가 없는 이미지를 학습한 classification모델이 몇장의 강아지 사진만으로 학습을 한 후 test data에 포함된 강아지 사진을 성공적으로 분류 할 수 있다.
* 한국 도로에서만 학습한 자율주행 모델이 미국 도로에서도 성공적인 운행이 가능하다.



 Meta Learining의 목표는 아래 식과 같다. Meta Learning은 레이블에 있는 데이터 
$$
D
$$
에 대하여 loss function을 최소화 할 수 있는 파라미터 $$ \theta^* $$를 찾겠다는 것이다. \
$$
\theta^* =arg\underset{\theta}min\mathbb{E}_{D\sim p(D)}
$$


 이는 언뜻 보기에 일반적인 machine learning과 유사해 보이지만 dataset자체가 하나의 data sample로서 사용된다는 점이 다르다. 













