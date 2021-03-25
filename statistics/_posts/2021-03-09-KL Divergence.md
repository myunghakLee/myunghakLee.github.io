---
layout: post 
title: "[Information Theory] KL Divergence"
# description: > 

---

KL Divergence에 대하여 설명하기 앞서 [cross entropy](https://myunghaklee.github.io/blog/statistics/2021-03-09-Information-theory_Cross_entropy/)를 보고 오는 것을 추천합니다.



# KL Divergence



 KL divergence는 두 확률분포의 차이를 계산하는데 사용되는 함수입니다. 딥러닝 모델을 만들 때를 예로 들면 우리가 가지고 있는 데이터의 분포 P(x)와 모델이 추정한 데이터의 분포 Q(x)간의 차이를 KLD(KL-Divergence)를 활용해 구할 수 있습니다. KLD의 식은 다음과 같습니다.

<img src="https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210325123152060.png?raw=tru" alt="image-20210325123152060" style="zoom:70%;" />

 만일 두 확률분포가 동일할 경우 KLD의 값은 0이 됩니다. 단 KLD는 비대칭으로 P와 Q의 위치가 바뀌면 KLD의 값도 바뀝니다. 즉 KLD를 거리함수라고 보면 안 됩니다.



 지금부터 위 수식에 대하여 자세히 알아보겠습니다. 위에서 KLD는 두 확률분포의 차이를 계산한다고 했는데 조금 더 정확히 말하면 KLD는 어떠한 확률 분포 P가 있을 때, 샘플링 과정에서 그 분포를 근사적으로 표현하는 확률분포 Q를 P대신 사용할 경우 엔트로피의 변화를 의미합니다. 따라서 우리는 식을 다음과 같이 고쳐 쓸 수 있습니다.

![image-20210325123242339](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210325123242339.png?raw=tru)

따라서 우리는 위 식을 통하여 두 확률 분포의 차이를 계산할 수 있습니다.





참고링크

https://ratsgo.github.io/statistics/2017/09/22/information/

https://ko.wikipedia.org/wiki/%EC%BF%A8%EB%B0%B1-%EB%9D%BC%EC%9D%B4%EB%B8%94%EB%9F%AC_%EB%B0%9C%EC%82%B0

