---
layout: page
title: "Topic & Message & Action"
description: > 
    본 포스팅은 Ros2에서 사용되는 Topic에 대해 간략히 정리하여 소개하고 있습니다.
    
---
* toc
{:toc}

## Topic이란?

**Topic**은 ROS(Robot Operation System)에서 로봇이나 프로그램이 서로 정보를 주고받을 때 사용하는 통신 채널이다. 마치 사람들이 대화할 때 소리나 문자로 메시지를 주고받는 것처럼, 로봇들도 주고받는 정보를 Topic을 통해 전송한다. Node 간의 통신을 쉽게 하기 위해서 ROS에서는 publisher와 subscriber라는 개념을 사용한다.

* Publisher: 특정 topic에 message를 보내는 node
* Subscriber: 특정 topic으로부터 message를 받는 node

아래 그림은 subscriber와 publisher가 서로 토픽을 주고받는 상황을 나타낸 것이다.

![alt text](/images/robotics/topic_message_action/image.png) [참조](https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.html)

Topic은 1:1 전송 외에도 1:N, N:1, N:N 전송도 지원한다.
![alt text](/images/robotics/topic_message_action/image-1.png)[참조](https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.html)

Topic은 비동기적으로 데이터를 주고받는다. publisher는 데이터를 지속적으로 발행하고 Subscriber는 그 데이터를 필요할 때 받아서 처리한다. 대표적인 예시로 센서 데이터가 있다.

## Service란?

**Service**는 두 node 간의 요청-응답 방식으로 통신하는 방법이다. 즉, 하나의 node가 특정 작업을 요청하고, 다른 node가 그 요청에 응답하는 구조다. 이는 topic과 달리 실시간으로 데이터를 주고받기보다는 특정 시점에 한 번 요청을 보내고 그에 대한 응답을 받는 방식이다.

Service는 topic과는 다르게 client가 호출할 때만 데이터를 제공한다.

![alt text](/images/robotics/topic_message_action/image-2.png) [참조](https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Services/Understanding-ROS2-Services.html)

또한, Service의 경우 topic과는 다르게 1:1 전송만 지원한다. 설령 많은 clients가 동일한 service를 사용하더라도 전송은 1:1로만 이루어진다.

![alt text](/images/robotics/topic_message_action/image-3.png) [참조](https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Services/Understanding-ROS2-Services.html)

Service는 동기적으로 데이터를 request-respond 방식으로 주고받는다. Publisher와 subscriber 구조를 이용하여 일방적인 데이터 흐름을 제공한다. 따라서 서비스는 한 번 요청과 그에 대한 응답으로 끝나는 단발성 작업에 적합하다. 즉, 요청에 대한 명확한 응답이 필요한 상황에 사용된다.

## Action이란?

**Action**은 장기 실행 작업을 위해 사용하는 것이다. 서비스는 요청-응답의 단순한 구조로 동작하는 반면, 액션은 시간이 오래 걸리는 작업(예: 로봇이 특정 위치로 이동하기)에도 적합한 통신 방식을 제공한다. 액션은 작업의 시작, 진행 중 상태, 완료 등의 여러 단계를 관리할 수 있어, 장시간 작업이나 복잡한 작업을 처리하는 데 유용하다.
Action은 아래 3가지 정보를 포함한다.

* Goal: client가 server에 보내는 목표 정보.
* Feedback: Server가 client에게 주기적으로 보내는 진행 상황
* Result: 작업이 완료되면 server가 보내는 최종 결과

![alt text](/images/robotics/topic_message_action/image-4.png)

예를 들면 로봇을 특정 위치로 이동시키라는 goal이 주어졌을 때 로봇은 목표로 이동하는 동안 계속해서 현재 위치를 client에게 feedback합니다. 그리고 로봇이 목표 지점에 도착하면 server가 작업 완료 결과를 client에게 보낸다.