---
layout: post
title: "[논문(번역)/classification, domain adaptation] Manifold Mixup Better Representations by Interpolating Hidden states"
#description: > 

---

### 



본 포스팅은 연구노트의 목적을 가지고 작성되었습니다.

참조 링크

https://arxiv.org/abs/1806.05236
https://github.com/vikasverma1077/manifold_mixup


**Abstract**

 Deep neural network는 training data를 학습하는 데에는 뛰어난 성능을 보이지만 약간 다른 test case를 평가할 때에는 종종 부정확한 결과가 나온다. 예를 들어 adversarial example, outliers, distribution shift 등이 있다. 이를 해결하기 위하여 이 논문에서는 Manifold Mixup을 제안한다. 이는 semantic interpolation을 추가 training signal로 사용하여 여러 level의 representation에서 더 부드러운 decision boundaries를 얻는다. 따라서 manifold mixup으로 훈련된 신경망은 분산 방향이 적은 class-representation을 학습한다.

그렇다면 우리는 왜 이러한 평탄화가 이상적인 조건에서 발생하는지에 대한 이론을 입증하고, 실제 상황에서 이를 검증하고, 이를 정보 이론 및 일반화에 관한 이전의 연구들과 연결할 것이다. 이는 많은 계산이 필요 없이 단 몇 줄의 코드로만 구현되었지만 manifold mixup은 supervised learning에 있어서 single-step adversarial attack에서의 견고성과 log-likelihood를 테스트할 때에 있어서의 baseline을 크게 향상시켰다.



**1. Introduction**

 신경망은 종종 training set에서는 잘 작동하지만 test set에서는 잘 작동하지 않는 문제가 있다. 신경망은 데이터가 distributional shifts의 대상이 될 수 있는 환경에 배포되고 있기 때문에 이는 큰 문제가 된다. 이에 대한 대표적인 예로 adversarial example이 있다. 신경망은 사람의 눈으로는 알아 볼 수 없는 perturbation(섭동)이 들어가는 경우 evaluation시에 문제가 생긴다. 이는 특히 보안에 민감한 응용 프로그램에 machine learning을 배포 할 때 심각한 위험으로 다가온다.

 우리는 SOTA 신경망의 hidden representations와 decision boundaries에 관한 몇가지 문제점을 깨달았다.

* 첫째, decision boundary가 종종 sharply하고 data에 지나치게 가깝다.

* 둘째, hidden representation space의 대부분은 manifold의 on/off에따라 confidence 값이 지나치게 높아진다.

 따라서 우리는 training example의 hidden representation에 대하여 신경망을 훈련시켜 위와 같은 문제를 해결하는 정규화 기법인 Manifold Mixup을 제안한다.

 Word embedding을 통한 analogy[[1\]](#_ftn1) 연구를 포함한 이전 연구는 interpolation이 combining factor를 하는데 효과적인 방법임을 보여주었다. High level의 representation은 종종 low-dimensional이며 선형 분류자에게 유용하기 때문에 hidden representation의 linear interpolation은 feature space의 의미 있는 영역을 효과적으로 탐색해야 한다. Hidden representation의 조합을 새로운 training signal로 사용하기 위하여 우리는 one-hot label 쌍에서 동일한 linear interpolation을 수행하여 soft target과 mixed example을 만든다.

![image](https://user-images.githubusercontent.com/12128784/106141872-1a689080-61b4-11eb-94d8-55178b746a5b.png)

 위 그림은 small data를 가지고 간단한 two-dimensional classification task에서의 manifold mixup의 영향을 보여주고 있다. 위 예의 (a)에서 보는 바와 같이 vanilla training은 irregular decision boundary와 (d)와 같은 hidden representation의 복잡한 배열로 이어진다. 또한 (a)와(d)에서는 데이터의 모든 지점을 높은 confidence를 갖는다고 판단한다.

이와 대조적으로 manifold mixup으로 동일한 deep neural network를 훈련하면 더 부드러운 decision boundary와 hidden representation의 더 간단한 (linear) arrangement로 이어진다. 요약하면 manifold mixup에 의하여 획득된 representation은 다음과 같은 2가지 특성(장점)을 갖는다.

* Class representation은 variation이 줄어드는 방향으로 더욱 flatten해진다.

* representation 사이의 모든 지점에는 낮은 confidence가 할당된다.

 보다 구체적으로 manifold mixup은 다음과 같은 이유로 deep neural network에서 일반화를 향상시킨다.

1. Multi level representation에서 training data를 더 멀리 떨어진 smooth한 decision boundary로 이어진다.(smoothness와 margin은 잘 알려진 generalization요소이다.)
2. 더 깊은 hidden layer의 interpolation을 활용하여 더 높은 level의 정보를 캡처하여 추가 training signal을 제공한다.
3. class representation을 flatten하게 하여 상당한 차이로 variance를 줄일 것이다.



다양한 실험을 통해 manifold mixup의 4가지 이점을 보여준다.

1. 더 나은 generalization(Cutout, Mixup, AdaMix, Dropout에 비하여, 아래 그림 참조)

2. Test sample에 대한 log-likelihood 개선

3. Predicted data에 대한 성능 향상

4. Adversarial attack에 대한 견고성 향상. 이는 manifold mixup이 decision boundary를 데이터로부터 멀리 밀어 냈다는 증거이다.(하지만 이는 decision boundary를 모든 방향으로 데이터로부터 멀어지게 하는 관점에서 정의되는 full adversarial robustness과 혼동해서는 안된다.)

![image](https://user-images.githubusercontent.com/12128784/106142214-8f3bca80-61b4-11eb-9627-6eb0477f2a9e.png)

**2. Manifold Mixup**

우선 f(x) = f~k~(g~k~(x))라는 식이 있다고 하자. 여기서 g­k는 입력 데이터를 layer k에서 hidden representation으로 mapping나는 신경망의 일부를 나타내고 fk는 이러한 hidden representation을 output f(x)에 부분 mapping한다. Manifold mixup을 사용하는 training f는 5단계로 수행된다. 

첫째, 신경망의 eligible layer set S에서 random layer k를 선택한다.

둘째, k 번째 layer에 도달할때까지 평소와 같이 2개의 임의 데이터 minibatch(x;y) 및 (x0;y0)를 제공한다.

셋째, 이러한 중간 minibatch에서 입력 mixup을 수행한다. 이는 mixed minibatch를 생성한다.

![image](https://user-images.githubusercontent.com/12128784/106143480-2b1a0600-61b6-11eb-8af4-5874e8e634c5.png)

넷째, 우리는 mixed minibatch인 ![image](https://user-images.githubusercontent.com/12128784/106144638-d4adc700-61b7-11eb-9e15-26ef2740eb27.png)를 사용한 output까지 network의 layer k에서 forward pass를 시킬 것이다.

다섯째, 위에서 나온 output은 신경망의 모든 매개변수를 업데이트하는 loss값과 기울기를 계산하는데 사용된다.

수식적으로 보면 manifold mixup은 다음을 최소화하는 방향으로 이루어진다.

![image](https://user-images.githubusercontent.com/12128784/106144743-f73fe000-61b7-11eb-9635-defadc0a952f.png)



------

[[1\]](#_ftnref1) 비유, 유추
