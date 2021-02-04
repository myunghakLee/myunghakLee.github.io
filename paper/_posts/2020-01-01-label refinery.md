---
layout: post 
title: "[논문(정리)/classification]Label Refinery: Improving ImageNet Classification through Label Progression"
# description: > 

---


참조 링크

https://arxiv.org/abs/1805.02641



본 포스팅은 연구노트의 목적을 가지고 작성되었습니다.





 

### **메인 아이디어 : label 을 hard하게 주지 말고 soft하게 주자.**

밑의 그림을 burrito라고 classification하는 것은 잘못되었다. 

Plate도 같이 찍혀 있으므로 label을 다음과 같이 soft하게 주는 것이 올바르다. 이 때 label을 soft하게 주기 위하여 teacher-student 모델이 사용된다.

 

**![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image002-1612438578288.png?raw=tru)**

다음 그래프를 보면 student의 성능이 teacher 못지 않는 성능일 보임을 알 수 있다.

![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image004-1612438578288.png?raw=tru)

 

 

수식은 다음과 같은 KL-divergence를 사용한다.

![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image006-1612438578288.png?raw=tru)

Student 모델인 Q(i)가 teacher 모델인 p(i)에 가까워질수록 KL-divergence는 작아진다.

 

이를 본 논문에 맞게 다시 쓰면 다음 식과 같은데 여기서 빨간 박스친 부분은 student model로부터 독립적이므로 식을 더 간단히 쓸 수 있다.

![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image008-1612438578288.png?raw=tru)

 

따라서 최종 식은 다음과 같이 나온다.

![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image010-1612438578288.png?raw=tru)

다음 결과를 보면 teacher-student 모델을 거칠수록 점차 accuracy가 올라가는 것을 볼 수 있다.

 

![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image012-1612438578288.png?raw=tru)

![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image014-1612438578288.png?raw=tru)

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 