---
layout: page
title: "ECE: Expected Calibration Error"
description: > 
    본 포스팅은 ECE의 정의에 대해서 다루고 있습니다..

---

* toc
{:toc}

Expected Calibration Error (ECE)는 기계 학습 모델의 예측 confidence(확신도)와 실제 정확성 사이의 불일치를 측정하는 평가지표입니다. 일반적으로 분류 문제에서 모델이 내놓는 확률 점수가 실제로 얼마나 정확한지를 평가하는 데 사용됩니다. 즉, 모델이 예측한 확률이 실제 사건의 발생 빈도와 얼마나 잘 일치하는지를 나타냅니다.

예를 들어 아래 그림과 같이 모델이 특정 사진에 대하여 강아지에 대한 confidence가 0.7이라면 실제로 모델이 정답을 맞추었을 확률도 0.7이어야한다.

![alt text](/images/etc/ece/image.png)

다시말해 모델이 confidence를 0.7이라고 예측한 사진을 모두 모아보면 그 사진들에 대한 정확도 역시 70%가 나와야 한다. 실제로는 조금 더 현실적으로 confidence를 0.7~0.8인 데이터를 모두 모았을 때 그 데이터에 대한 정확도가 75%와 얼마나 차이나는 지를 계산한다. 이를 수식으로 쓰면 다음과 같다.

$$
    ECE =  \sum_{b=1}^{B} {\frac{\left| b \right|}{N} |acc(b) - conf(b)| }
    \\[1em]
    conf(b) = {\frac{1}{\left| b \right|}}\sum_{j \in b} p_{j}
    \\[1em]
    acc\left(b\right) = {\frac{1}{|b|}}\sum_{j \in b} \Bbb{1}(p_j = y_j)
$$










