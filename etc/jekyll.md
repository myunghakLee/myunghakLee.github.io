---
layout: page
title: Jekyll(hydejack)
description: > 
        Jekyll의 hydejack theme의 간단한 사용법에 대한 설명을 진행하는 포스팅입니다.

---
* toc
{:toc}


## 설치 방법


1. [링크](https://hydejack.com/download/)를 클릭하여 테마를 다운받습니다.
2. 자신의 로컬 환경에 새로운 폴더를 만들고 자신의 github repository에 연결합니다.
  ~~~console
  // file: "console"
# 새폴더 만들고 폴더에 들어가기
mkdir MyBlog
cd MyBlog
# github repository에 연결하고 pull하기
git init
git remote add origin https://github.com/****/****.github.io
git pull origin main
  ~~~

3. 폴더내 모든 데이터 삭제
  ~~~console
  // file: "console"
rm -r ./*
  ~~~

4. Jekyll의 기본 패키지 다운
  ~~~console
  // file: "console"
jekyll new ./
  ~~~

5. 앞서 받은 테마를 현재 폴더에 복붙
  ~~~console
  // file: "console"
cp -r ../{Hydejack 테마 폴더}/* ./
  ~~~

6. 충돌나는 파일 제거
  ~~~console
  // file: "console"
rm 404.html about.markdown index.markdown
  ~~~

7. bundel 설치 및 서버 실행
  ~~~console
  // file: "console"
bundle install
bundle exec jekyll serve
  ~~~

8. _config.yml파일 수정
  
  _config.yml파일 내부에 아래 그림과 같은 부분에서  theme: jekyll-theme-hydejack을 주석처리하고 그 밑에줄의 주석을 제거
  ![alt text](/images/etc/jekyll/image.png)


## SideBar에 카테고리 추가

왼쪽 SideBar에 카테고리를 추가하려면 <span style="background-color:#fff5b1">_config.yml</span> 파일에서 아래 그림과 같은 부분을 찾아 **title**과 **url**부분을 채워주면 됩니다.

![alt text](/images/etc/jekyll/image-1.png){:.centered}

그 후 새 폴더를 url과 같은 이름으로 만든 후 안에 <span style="background-color:#fff5b1">README.md</span>를 작성하면 됩니다.
~~~console
// file: "README.md의 예시"
---
title: Etc
description: >
  해당 페이지에서는 대분류에 속하지 못한 기타 머신러닝 관련 정보에 대한 포스팅을 진행하겠습니다.
hide_description: false
sitemap: false
permalink: /etc/
---


## Machine Learning
* [Link Prediction(링크 예측)]{:.heading.flip-title} --- Link Prediction이란.
* [Knowledge Graph(지식 그래프)]{:.heading.flip-title} --- Link Prediction이란.


## Paper

[Link Prediction(링크 예측)]: link-prediction
[Knowledge Graph(지식 그래프)]: knowledge-graph
[Jekyll(hydejack)]: jekyll

## Etc
* [Jekyll(hydejack)]{:.heading.flip-title} --- Jekyll(hydejack)의 간단한 사용법
~~~


그 후 해당 폴더내에 계속해서 markdown 파일을 추가해주면 됩니다.