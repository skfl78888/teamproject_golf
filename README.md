0. TODO: extensions.list 사용해서 extension 동기화 할것!
    - vscode powershell에서 실행할것!(카톡으로 보내준 extensions.list 사용)
    - 참고 사이트: https://oysu.tistory.com/52

1. 로컬서버 아나콘다 환경에서 새로운 가상환경 생성
  - 로컬서버 연결 후 까만창에다가 [conda create --name TeamProject python=3.10]

2. vscode를 통해 로컬서버 원격으로 연결(까만창에서 로컬 서버를 연결하는 과정을 vscode 에디터로 하는 과정)
  - 참고 사이트:  https://doheejin.github.io/vscode/2021/02/25/vscode-server.html

3. 로컬서버에 git설치
  - [sudo apt-get install git] 명령어를 입력하여 패키지 리스트를 업데이트합니다.
  - [sudo apt install git] 명령어를 입력하여 깃을 설치합니다.
  - 참고 사이트: https://coding-factory.tistory.com/502

4. github 설정
  - github 가입
  - 터미널에서 git config --global user.name [~~]
  - 터미널에서 git config --global user.email [~~]
  - 충호한테 git 콜라보레이터 설정해달라고 하기


5. 충호의 git golf 레포 참고해서 git clone(충호가 만들어 놓은 폴더를 내려 받는 과정)
  - 사이트: https://github.com/skfl78888/teamproject_golf.git

6. clone한 폴더 열기

7. github에 등록된 req~~.txt 파일 사용해서 패키지 설치
  - 터미널에서 [conda activate TeamProject]
  - [pip install -r requirments.txt 실행]

8. git remote 설정(로컬 저장소와 원격 저장소 연결해주는 과정)
  - 터미널에서 [git remote add golf https://github.com/skfl78888/teamproject_golf.git]

![address](https://user-images.githubusercontent.com/68595420/206379446-75638f19-edb3-4362-ae7c-47a6ac743f29.jpg)