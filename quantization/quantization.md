---
layout: page
title: Quantization이란
description: > 
    본 포스팅은 Quantization에 대해 간략히 정리하여 소개하고 있습니다.
    
---

* toc
{:toc}



Quantization은 경량화 기법중 하나로 신경 망을 구성하는 파라미터나 activation을 더 적은 비트수로 표현하여 추론 속도를 높이는 과업입니다. 일반적으로 float32 형태의 자료형으로 표현되는 값을 INT8이나 INT4 등으로 표현합니다. 아래는 quantization 논문에서 일반적으로 사용되는 표기법입니다.

* W8A32: weight를 8비트로, activation을 32비트로 표현(quantization을 진행하지 않음)
* W8A8: weight를 8비트로, activation을 8비트로 표현
* W4A8: weight를 4비트로, activation을 8비트로 표현

![alt text](/images/quantization/quantization/image.png)

## Quantization의 장점

Quantization을 사용하면 아래와 같은 측면에서 장점이 존재합니다.

* 모델 크기 감소

    * 더 작은 비트수를 사용하여 가중치와 활성화를 표현함으로써 메모리 사용량과 저장 공간 절약

* 연산 속도 향상

    * 정수 연산을 사용함으로 인하여 부동 소수점보다 연산이 빠르며 하드웨어에서 더 효율적으로 실행

* 전력 소비 감소

    * 적은 연산으로 전력 소모 감소
    * 모바일 및 임베디드 장치에서 배터리 수명 연장

* 경량화된 모델 배포

    * 모바일 및 IoT 디바이스와 같은 제한된 환경에서 모델 배포 용이
    * 클라우드 비용 절감

* 인프라 비용 절감

    * 작은 모델 크기로 저장소 및 네트워크 비용 절감
    * 효율적인 자원 활용으로 비용 절감

