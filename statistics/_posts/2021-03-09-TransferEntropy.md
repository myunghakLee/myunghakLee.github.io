---
layout: post 
title: "[Information Theory] Transfer Entropy"
# description: > 

---

Transfer Entropy를 공부하기 위해서는[Information theory](https://myunghaklee.github.io/blog/statistics/2021-03-08-Information-theory/) [cross entropy](https://myunghaklee.github.io/blog/statistics/2021-03-09-Information-theory_Cross_entropy/) [KL - Divergence](https://myunghaklee.github.io/blog/statistics/2021-03-09-KL-Divergence/)개념이 필요함을 알려드립니다.



# Transfer Entropy



 Transfer Entropy는 두 객체의 인과관계의 정도를 알아보기 위하여 도입된 개념이다. 즉 Transfer entropy에서는 시간의 개념이 포함되어 있다. 

만약 X의 행동이 Y로부터 유발된 것이라고 추측한다면 X의 현재 행동은 Y의 행동과 상관관계가 존재하야 한다. 즉 x~t~는 y~t-1~, y~t-2~ … 와 상관관계가 존재해야 한다는 것이다. 

 이 말은 X의 이전 data만 주어졌을 때와 여기에 Y의 data가 추가로 주어졌을 때 불확실성(entropy)이 변한다는 것이고 이 차이가 클수록 x~t~ 는 Y의 이전 값들에 의해 영향을 크게 받는다는 것이다.

<img src="https://user-images.githubusercontent.com/12128784/112811181-c2b4ab80-90b6-11eb-910a-f158d5ed4460.png" alt="image" style="zoom:80%;" />

위 식을 다시 설명하면 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image002-1616644042471.png?raw=tru?raw=tru)이 주어졌을 때 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image004-1616644042471.png?raw=tru?raw=tru)이 갖는 정보량과  ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image002-1616644042471.png?raw=tru?raw=tru)와 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image006-1616644042471.png?raw=tru?raw=tru)이 주어졌을 때 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image002-1616644826933.png?raw=tru)이 갖는 정보량의 차이를 구하겠다는 식이다. 만약 이 값의 차이가 크다면 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image002-1616644780859.png?raw=tru)이 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image002-1616644870316.png?raw=tru)에 미치는 영향이 크다고 봐도 될 것이다.

 

#### Example

극단적으로 말해서 만일 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image008-1616642648200.png?raw=tru?raw=tru)이지만 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image002-1616644278564.png?raw=tru?raw=tru)이라면 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image004-1616644278564.png?raw=tru?raw=tru)은 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image006-1616644278564.png?raw=tru?raw=tru)이 주어져야 비로소 확정되고 이는 곧 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image002-1616644515952.png?raw=tru?raw=tru)은 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image004-1616644515952.png?raw=tru?raw=tru)에 지대한 영향을 끼친다는 것이다.

 

참조 링크: https://mons1220.tistory.com/154

