---
layout: post 
title: "[논문(정리)/Motion Prediction] Learning Lane Graph Representations for Motion Forecasting - ECCV2020 Oral"
# description: > 

---





### Abstract

본 논문에서는 actor-actor, actor-lane, lane-actor, lane-lane의 복잡한 interaction을 포착하는 방법을 제시한다. LaneGCN과 actor-map interaction으로 구동 되며 이는 정확하고 사실적인 multi-modal trajectory prediction이 가능하게 해준다.



### 1. Introduction

  Actor의 움직임은 대부분 차선에 종속적이다. 따라서 차선 정보를 신경망의 입력으로 넣어 주는 것이 유리하다. 최근 신경망에서는 지도 정보를 rasterize하여 집어넣는 시도가 있었다. 그러나 이러한 방법은 다음과 같은 단점이 있다.

1. 정보의 손실을 피할 수 없다

2. 지도는 복잡한 topology의 그래프로 구성되어 있는데 여기에 2D convolution을 적용하는 것은 비 효율적이다(보통 차선의 경우 긴 범위를 갖는데 이를 모두 포착하려면 너무 큰 receptive field가 필요하다).

3. 차선의 경우 거리는 가깝더라도 동일한 방향의 차선과 반대편의 차선은 전혀 다른 의미를 가지고 있다.



따라서 본 논문은 다음과 같은 방법을 제시한다.

1. Rasterization 대신 vectorized map으로부터 lane graph를 구성하여 정보 손실을 피한다. 그리고 여기에 lane graph의 복잡한 topology와 long range dependencies를 효과적으로 포착하는 LaneGCN(Lane Graph Convolution Network)을 적용한다.

2. Actor와 lane을 모두 node로 표기한다. 그 후 Lane과 actor에서 각각 feature를 뽑기 위하여 1D CNN과 LaneGCN을 이용한다. 그 후 spatial attention과 다른 LaneGCN을 활용하여 4가지 유형의 상호작용을 모델링한다(actor-to-lane, lane-to-lane, lane-to-actor and actor-to-actor).

3. 이와 같은 방법으로 Argoverse dataset에서 SOTA를 찍었다.

![image-20210406121128163](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406121128163.png?raw=tru?raw=tru)



### 2.Related Work

  본 논문의 Map representation은 VectorNet과 유사하나 VectorNet은 lane을 node로 본 반면 본 논문은 lane segment를 node로 보았다. 즉 resolution이 조금 더 높다. 또한 본 논문은 GCN에서 영감을 얻어 lane graph용으로 설계된 특화된 버전인 LaneGCN을 제안한다. 또한 모델에 lane-graph의 복잡한 topology와 long-range dependencies를 포착하는데 효과적인 multi scale dilated convolution을 사용하였다.



### 3.Lane Graph Representations for Motion Forecasting

 이번 장에서는 traffic actor와 HD map의 fusion에 대하여 설명하겠다. 본 논문에서는 다음 4가지 module을 사용하였다.



1. **ActorNet**: Actor의 past trajectory를 input으로 받아와 1D convolution으로 feature를 추출한다.

2.   **MapNet**: HD map으로부터 lane graph를 구성하고 LaneGCN을 이용하여 lane node feature를 뽑는다.

3. **FusionNet**: 4개의 interaction block으로 구성되어 있다(lane to lane block에는 another LaneGCN을 이용하는 반면 다른 block에는 spatial attention layer를 사용한다).

   ​	A.   Actor to lane block: actor node에서 lane node로 실시간 교통 정보를 fusion

   ​	B.   Lane to lane block: lane graph를 통해 정보를 전파, lane feature를 update

   ​	C.   Lane to actor block: lane node에서 actor node로 update된 정보 전파

   ​	D.   Actor to actor block은 actor간의 상호작용을 수행

4. **Prediction** **Header**: fusion 후 actor feature를 사용하여 multi modal trajectory를 생성

![image-20210406121426808](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406121426808.png?raw=tru?raw=tru)



#### 3.1 ActorNet: Extracting Traffic Participant Representations

 각 trajectory는 각 sequence의 변위로 나타낸다. ∆p~t~는 time step t-1에서 t까지의 변위이다. 그리고 T는 trajectory의 크기이다. 만약 trajectory가 T보다 작으면 0으로 padding해준다. 그리고 각 step이 padding된 것인지 아닌지를 나타내는 1*T크기의 binary mask를 추가해준다.

![image-20210406121558183](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406121558183.png?raw=tru?raw=tru)



 CNN과 RNN 모두 temporal data에 사용된다. 그러나 본 논문은 1D CNN을 사용하였다. Actor Net의 output은 temporal feature map이고 t=0은 actor feature로 사용된다. 본 논문의 network는 1D convolution의 3 group/scale을 가진다. 각 group은 첫 번째 block의 stride를 2로 하는 2개의 residual block으로 구성된다. 그 후 FPN(Feature Pyramid Network)를 사용하여 multi-scale feature를 fusion하고 또 다른 residual block을 적용하여 output을 얻어낸다.

<img src="https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406160249565.png?raw=tru?raw=tru" alt="image-20210406160249565" style="zoom:50%;" />

#### **3.2 MapNet: Extracting Structure Map Representation**

앞서 설명했던 대로 Map을 rasterize해서 집어넣으면 문제가 발생한다. 따라서 본 논문에서는 graph를 이용하였다.

**Map Data**: map data를 set of lanes and their connectivity로 나타냈다. 바로 붙어있는(반대편 차선등은 제외) 2 lane은 4가지 type이 있다(predecessor, successor, left neighbor, right neighbor).

**Lane Graph Construction**:본 논문에서는 lane node를 centerline의 두 연속된 점(그림 3의 회색 원)에 의해 형성된 직선 segment로 정의한다. Lane node의 위치는 두 끝점의 평균 좌표이다. 위에서 설명했듯 Lane centerline간의 연결에 따라 lane node는 총 4가지로 나뉜다. 본 논문에서는 lane node를 V ∈ R^N×2^로 표시한다. 여기서 N은 lane node의 수이다. 또한 A~i~ ∈ R^N×N^는 4 개의 인접 행렬 {A~i~}~i∈{pre,suc,left,right}~와의 연결성을 나타낸다. 즉 노드 k가 노드 j의 i형 이웃이면 A~i,j,k~ = 1이다. 즉 “node k는 i-type을 갖고 j의 이웃이다”라는 의미이다. 이러한 전제 하에서 만일 A가 B의 Successor라면 A에서 B로의 접근은 가능하나 B에서 A로의 접근은 불가능하다. 그러나 만약 둘의 관계가 neighbor라면 양쪽모두에서 접근이 가능하다.

<img src="https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406160525989.png?raw=tru?raw=tru" alt="image-20210406160525989" style="zoom:40%;" />



**LaneConv Operator**: 

 보통 graph convolution의 경우 Y = LXW이다. 여기서 X ∈ R^N×F^는 node feature이고 W ∈ R^F×O^는 weight matrix, Y ∈ R^N×O^는 output이며 L ∈ R^N×N^은 Laplacian matrix이며 다음 식으로 얻어진다.





1. 어떤 node feature가 lane graph의 정보를 보존시킬지 명확치 않음

2.   Single graph Laplacian은 connection type을 포착할 수 없다. 즉 차선의 방향 정보를 잃는다.

3. 이러한 형태의 graph convolution은 long range dependencies(dilated convolution같은)를 처리하는 것이 어렵다.

따라서 본 논문은 위의 단점을 해결한 LaneConv라는 새로운 연산자를 소개한다.

<img src="https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406161306592.png?raw=tru?raw=tru" alt="image-20210406161306592" style="zoom:35%;" />

  여기서 I는 Self-connection을 그리고 A는 different node와의 connection을 뜻한다. 그리고 D는 output을 normalize하는 역할이다. 그러나이러한 vanilla graph convolution은 motion prediciton에서는 비 효율적이다. 왜냐하면 lane graph의 information이 보존된다는 보장도 없으며 connection type 역시 포착하지 못한다. 그리고 long range dependency 역시 포착하지 못한다. 따라서 본 논문은 LaneConv를 제시한다.



  *Node feature*: 본 논문에서는 lane node의 information을 enccoding하기 위하여 lane의 모양(크기 및 방향)과 위치모두를 고려하였다.  v~i~는 lane의 중심 좌표이고 v~i~^end^및 v~i~^start^는 마지막 및 처음 좌표이다. 그리고 이렇게 encoding된 x~i~를 row로 하는 matrix가 X이다.

![image-20210406161435791](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406161435791.png?raw=tru?raw=tru)



  *LaneConv*: node feature는 오직 local informatio만 포착한다. 따라서 이를 aggregate해주기 위하여 다음과 같은 LaneConv식을 이용하였다.아래 식에서 A~i~와 W~i~는 adjacency와 weight matrix이다. 그리고 이때 A~suc~와 A~pre~는 단위 행렬을 오른쪽 위(non-zero superdiagonal)와 왼쪽 아래(non-zero subdiagonal)로 한칸 이동하여 얻은 행렬이다.

<img src="https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406161543606.png?raw=tru?raw=tru" alt="image-20210406161543606" style="zoom:100%;" />



*Dilated LaneConv*: actor의 속력이 빠른 경우 모델은 long range dependency를 포착할 필요가 있다. 보통의 grid graph의 경우 dilated convolution operator가 효율적으로 long range dependency를 포착할 수 있다. 여기에서 영감을 받아 본 논문에서는 dilated LaneConv operator를 제안한다.아래 식에서 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image004-1617695386558.png?raw=tru?raw=tru)는 A~pre~의 k제곱이다. 이는 information을 k-step으로 propagate 가능하게 해준다. 이 때 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/1617697074_6849957/clip_image004.png?raw=tru?raw=tru)는 굉장히 sparse하기 때문에 sparse matrix multiplication으로 효율적인 계산이 가능하다. 또한 extended LaneConv operator는 predecessor및 successor에서만 사용한다(long range dependency는 차선 방향을 따르기 때문).

![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image002-1617695386557.png?raw=tru?raw=tru)



**LaneGCN**: 본 논문에서는 multi-scale LaneConv operator를 사용하여 LaneGCN을 구축한다. 2번식과 3번식을 사용하여 C dilation size를 가진 multi-scale LaneConv operator를 얻는다.

![image-20210406165047618](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406165047618.png?raw=tru?raw=tru)

![image-20210406165110245](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406165110245.png?raw=tru?raw=tru)









#### 3.3 FusionNet

  본 section에서는 ActorNet과 MapNet으로부터 얻어진 actor와 lane node의 정보를 fuse할 것이다. 이전 연구들이 Actor간의 interaction은 포착을 하였지만 Actor와 map사이의 종속성은 제대로 파악하지 못하였다. 하지만 본 논문은 이를 가능하도록 하였다.

본 논문에서는 4개의 Fusion module이 있다.

1. A2L(actors to lanes): real-time traffic information to lane nodes(blockage or lane의 사용중 여부)

2. L2L(lanes to lanes): lane graph를 이용하여 traffic information을 propagate하여 lane node feature를 update한다.

3.   L2A(lanes to actors): update된 map feature를 fuse하여 actor로 보냄

4. A2A(actors to actors): motion forecasting을 위한 prediction header를 생성하는데 사용되는 Actor가 interaction을 처리하고 output actor feature를 생성.



  L2L은 MapNet에서 사용된 것과 동일한 architecture를 가진 다른 LaneGCN을 사용하여 구현된다. 그리고 A2L, L2A, A2A에 대하여 special attention layer를 사용한다. Attention layer는 3모듈에 동일한 방식으로 적용된다. 예를 들어 A2L은 actor node i가 주어지면 다음과 같이 context lane-node j에서 feature를 aggregate한다.

![image-20210406165301740](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406165301740.png?raw=tru?raw=tru)

 위 식에서 x~i~는 i번째 node feature이고 φ는 layer normalization과 ReLU이다. 그리고 ∆~i,j~는 MLP(v~j~-v~i~)이고 여기서 v는 node location이다. 또한 context node는 actor node i의 L2 distance가 특정 threshold보다 작은 것들이다(본 논문에서는 A2L, L2A, A2A에 각각 7,6,100 미터를 사용). 그리고 A2L, L2A, A2A는 모두 attention layer와 linear layer 및 residual connection으로 구성된 2 residual block으로 구성된다.

<img src="https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406165512827.png?raw=tru?raw=tru" alt="image-20210406165512827" style="zoom:60%;" />



#### 3.4 Prediction Header

  After fusion actor feature를 입력으로 사용하여 multi modal prediction header가 최종 motion prediction을 출력한다. 각 actor에 대하여 K개의 future trajectory와 confidence score를 예측한다. header에는 각 mode의 trajectory를 예측하는 regression branch와 각 mode의 confidence를 예측하는 classification branch가 있다. m번째 actor의 경우 regression branch에 residual block과 linear layer를 적용하여 BEV coordinate의 K sequence를 regress한다.

![image-20210406165717224](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406165717224.png?raw=tru?raw=tru)

  위 식에서 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image002-1617695846222.png?raw=tru?raw=tru)는 m번째 actor의 k-th mode BEV 좌표이다. Classification branch에서는 K distance embedding을 위하여 ![img](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/clip_image004-1617695846223.png?raw=tru?raw=tru)에 MLP를 적용한다. 그 후 각각의 distance embedding을 actor feature와 concatenate하고 residual block과 linear layer를 적용하여 K confidence score를 출력한다.

![image-20210406165746107](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406165746107.png?raw=tru?raw=tru)





#### Summary

지금까지의 것들을 다 합치면 다음과 같다.

![image-20210406165855896](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406165855896.png?raw=tru?raw=tru)





### **Experimental Evaluation**

본 논문은 [Argoverse Dataset](https://www.argoverse.org/)에 대한 평가지표를 구하였고 SOTA를 찍었다(그러나 2021년 4월 기준 밀려남).

![image-20210406170022574](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406170022574.png?raw=tru?raw=tru)







**Importance of each module**: 아래 표에서는 ActorNet에 여러 모듈들을 추가해가며 실험을 한 결과이다. 그 결과 본 논문에서는 다음과 같은 결과를 도출해 냈다.

1. 모든 모듈이 성능향상에 기여하였다.

2.   actor에서 map으로의 정보 전달이 효율적인 motion forecasting preformance에 큰 영향을 주는 traffic information을 가져오는데 도움이 됐다.

3. ActorNet에 A2A만 추가하는 것에 비하여 L2A, A2L, L2L을 모두 추가하는 것이 더 좋은 결과가 나오는 것을 보아 L2A, A2L, L2L모두 각 factor간의 interaction을 용이하게 하는데 이용될 수 있는 것을 알 수 있다.

![image-20210406170232676](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406170232676.png?raw=tru?raw=tru)





**Lane Graph Operators**: 아래 표는 vanilla graph convolution에 각각의 component를 추가함에 따라 얻는 효과에 대한 실험결과이다. 아래 표에서 보면 모든 component가 성능을 굉장히 향상 시키는것을 볼 수 있다. 특히 residual block은 단 7%의 파라미터 추가만으로 굉장히 큰 성능 향상을 가져왔다.

![image-20210406170343163](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406170343163.png?raw=tru?raw=tru)

**Qualitative Results**: 아래 그림에서는 여러 상황에서 본 논문의 모델과 다른 모델의 차이를 보여준다.

  첫번째 행에서는 우회전하는 시나리오이다. 다른 방법들은 우회전을 제대로 포착하지 못하였으나 본 논문의 모델은 가능했다.

  두번째 행은 agent가 비보호 좌회전을 위해 대기중인 경우이다. 이는 history가 부족하여 지도 정보를 포착하는 것이 필수적이다. 따라서 다른 모델들은 제대로 된 예측을 하지 못하였고 본 논문의 모델은 성공적인 예측을 보였다.

  세번째 행은 자동차가 감속하여 교차로에 정지하는 경우이다. 이 경우 본 논문의 모델이 다른 모델에 비하여 더 많은 감속 mode를 생성하였다.

  네번째 행은 급격한 가속을 한 경우인데 이 경우 어떠한 모델도 제대로 포착을 하지 못하였다. 이는 예측을 하기위한 충분한 정보가 없기 때문일 가능성이 크다.

![image-20210406170451581](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406170451581.png?raw=tru?raw=tru)





### **5. Conclusion**

본 논문에서는 lane graph representation을 학습하고 완전한 actor-map interaction을 수행하기 위한 prediction model을 제안한다. 특히 map을 집어넣을 때 rasterize하지 않고 vector로 바꾸어 lane graph를 구성하고 LaneGCN을 이용하여 topology feature를 추출한다. 또한 spatial attention과 LaneGCN을 이용하여 actor와 lane의 정보를 fuse하여 대규모 Argoverse dataset으로 실험을 진행하였고 SOTA를 찍었다.





### **Appendix**

다음은 모델의 세부 architecture이다. 본 모델은 4개의 모듈로 구성되어 있다.

ActorNet은 1D CNN으로 시간적 feature를 추출하고 multi scale feature를 FPN과 병합한다. MapNet은 LGC(Lane Graph Network)으로 lane topology를 추출한다. LGN은 4개의 multi-scale LaneConv residual block이다. Actor-map fusion cycle은 A2L, L2L, L2A, A2A로 이루어진 fusion network stack이다. A2L, L2A, A2A는 2개의 attention residual block의 stack이며 L2L은 또다른 LGN이다. 마지막으로 update된 actor feature는 multi-mode trajectory 및 confidence score를 생성하기 위해 prediction header에서 사용된다.

![image-20210406170650334](https://github.com/myunghakLee/GIT_BLOG_IMAGE/blob/master/image-20210406170650334.png?raw=tru?raw=tru)