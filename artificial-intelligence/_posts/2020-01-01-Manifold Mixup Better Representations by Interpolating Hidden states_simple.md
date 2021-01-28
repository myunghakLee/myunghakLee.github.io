---
layout: post
title: [논문(정리)/classification, domain adaptation] Manifold Mixup Better Representations by Interpolating Hidden states
# description: > 
https://arxiv.org/abs/1806.05236
https://github.com/vikasverma1077/manifold_mixup
---

### 

Manifold Mixup전에 Mixup(https://arxiv.org/abs/1710.09412)부터 다루고 넘어가자.

에서 Mixup은 DataAugmentation의 방법중 하나이다. 원래라면 classification시 다음과 같이 예측하는것이 맞다.

![image](https://user-images.githubusercontent.com/12128784/106145317-bac0b400-61b8-11eb-9792-e631c289ff00.png)

하지만 Mixup 논문에서는 개와 고양이를 섞은 그림에서는 다음과 같이 예측하는 것이 맞다고 주장한다.

![image](https://user-images.githubusercontent.com/12128784/106145388-d5932880-61b8-11eb-855f-6f24f65ccac7.png)

![image](https://user-images.githubusercontent.com/12128784/106145401-d9bf4600-61b8-11eb-91f3-f13b46c57eb8.png)



즉 Mixup은 레이블 값을 0 아니면 1로 보지 않는다. 그렇다면 Manifold Mixup에 대해 알아보자.

Mixup은 input data를 mix했다면 Manifold Mixup은 Manifold 상에서 Mix를 한다. 여기서 Manifold란 데이터가 머물고 있는 공간 즉 hidden represetntation이라고 보면 된다. 전형적인 classfication문제에서는 다음과 같이 군집화가 이루어질 것이다.

![image](https://user-images.githubusercontent.com/12128784/106145575-10955c00-61b9-11eb-8f80-499f232edeb8.png)

그렇다면 원과 사각형이 적당히 섞인 도형을 input으로 집어 넣으면 이 도형은 다음과 같이 hidden space에서 중간위치에 있을 것이다.

![image](https://user-images.githubusercontent.com/12128784/106145897-679b3100-61b9-11eb-942b-ea9deba09f36.png)

즉 Mixup은 Results만을 가지고 추측을 하였다면 Manifold Mixup은 중간 결과물, 즉 layer단위로 판단하겠다는 것이다.

이러한 Manifold Mixup을 사용하면 다음과 같이 정규화 효과를 얻을 수 있다(곡선이 좀더 매끄러워 짐).

![image](https://user-images.githubusercontent.com/12128784/106146012-8994b380-61b9-11eb-9a1f-cadc5353f79a.png)

그리고 이러한 결과는 다른 유명한 정규화 기법의 성능에 준하는 결과를 보인다.

![image](https://user-images.githubusercontent.com/12128784/106146092-9e714700-61b9-11eb-91ae-52575f7b23d6.png)

또한 이러한 정규화 뿐 아니라 Adversarial training의 효과도 갖는다.