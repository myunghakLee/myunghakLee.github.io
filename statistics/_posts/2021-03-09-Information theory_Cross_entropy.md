---
layout: post 
title: "[Information Theory] Cross Entropy"
# description: > 

---



# Information Theory



## Cross entropy

 우선 Cross entropy를 들어가기 엔트로피를 구하는 식을 알아보자. 엔트로피를 구하는 식은 다음과 같다. 이에 관한 자세한 설명은 다음 링크를 참조하자. 

[Entropy]: https://myunghaklee.github.io/blog/statistics/2021-03-08-Information-theory

![image-20210309144934053](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210309144934053.png?raw=tru)

확률 분포 P와 Q가 있다고 가정하자. P = [0.25, 0.25, 0.25, 0.25]가 되고 Q = [0.125, 0.125, 0.25, 0.5]라고 가정하자. 이 때 확률분포 Q를 P대신 사용할 경우 엔트로피의 변화량을 구하고 싶을 경우 다음과 같은 식을 사용한다.

![image-20210309144629145](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210309144629145.png?raw=tru)

 여기서 우리가 유념해둘 것은 Q를 P대신 사용했으므로 정보량은 Q에 해당하는 것을 사용한다. 그러나 그 정보량이 나올 확률은 변하지 않으므로 정보량의 기대값 즉 entropy는 위 식과 같이 구하는 것이다. 이제 위 식을 일반화하여 적어보면 다음과 같다.

![image-20210309145129238](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210309145129238.png?raw=tru)

 보통 머신러닝의 경우 P가 ground truth가 되고 Q가 현재 학습한 확률값이다. 그리고 우리가 학습한 확률값 Q가 P에 가까워질수록 cross entropy의 값은 작아지게 된다. 따라서 loss 값을 줄이려는 머신러닝의 특성상 cross entropy를 loss function으로 사용하면 학습이 성공적으로 진행될 수 있는 것이다.

