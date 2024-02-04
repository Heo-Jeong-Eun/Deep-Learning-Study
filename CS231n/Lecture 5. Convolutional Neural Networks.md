## **Lecture 5. Convolutional Neural Networks**

### **Convolutional Neural Network**

> **Fully Connected Layer**가 하는 일은 어떤 Vector를 가지고 연산을 하는 것이다. 또한 **Activation은 이 Layer의 출력**이다.
> 
> 
> <img width="637" alt="스크린샷 2024-02-04 오전 2 12 11" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/60a5e553-c5cd-41c5-ba5f-a67d8563b34a">
> 
> 위 예시를 보면 입력 값으로 32 X 32 X 3의 이미지가 있고, 이 이미지를 길게 펴 3072 차원의 Vector로 만든다. 
> 
> 그리고 가중치 W, 10 X 3072 행렬을 Vector와 곱하고 Activation을 이 Layer의 출력값으로 얻는다. 위 예시의 경우 10개의 출력 값을 가진다. 
> 
> <img width="637" alt="스크린샷 2024-02-04 오전 2 11 24" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/6594e7f1-300c-41fd-8f0d-7fb2210be28a">
> 
> **Convolution Layer는 기존의 구조를 보존**시키고, **Fully Connected Layer는 보존하지 않는다.**  
> 
> 32 X 32 X 3의 이미지를 받아 하나의 긴 Vector로 늘리는 것이 아니라 기존의 이미지 구조를 그대로 유지한다. 
> 
> 그리고 5 X 5 X 3 **Filter가 가중치**가 된다. 이 Filter를 가지고 이미지 슬라이딩을 하며 공간적으로 내적을 수행하게 된다. 
> 
> <img width="700" alt="스크린샷 2024-02-02 오후 9 29 03" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/a40fbb0a-c091-4406-b4e7-2fb74f723e83">
> 
> Filter는 이미지의 깊이만큼 확장되지만, 이미지의 아주 작은 부분만 취할 수 있다. → 32 X 32 중 5 X 5에 해당하는 부분, Depth는 3
> 
> Filter를 이미지의 어떤 공간에 겹쳐두고, Filter의 각 W와 이에 해당하는 이미지의 픽셀을 곱해준다. **기본적으로 W^tx + b를 수행**하는 것이다. 
> 
> Filter가 슬라이딩 할 때, **Convolution은 이미지의 좌상단부터 시작**한다. 
> 
> Filter의 모든 요소를 가지고 **내적을 수행**하게 되면 하나의 값을 얻게 되고 **계속 슬라이딩** 한다. 
> 
> Conv 연산을 수행하는 값들은 다시 출력 Activation Map에 해당하는 위치에 저장한다. 아래 예시를 보면 **입력 이미지와 출력 Activation Map의 차원이 다르다**는 것을 알 수 있다. → 입력 32 X 32, 출력 28 X 28
> 
> <img width="700" alt="스크린샷 2024-02-02 오후 9 36 33" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/6f8936c0-2634-4a95-aef1-b8a4b0ede873">
> 
> 보통 Convolution Layer에서는 여러 개의 Filter를 사용하는데, Filter마다 다른 특징을 추출하고 싶기 때문이다. 
> 
> 아래 예시의 두 번째 Filter인 초록색의 5 X 5 X Filter를 연산하고 나면 앞서 계산했던 Activation Map과 같은 크기의 새로운 Map이 만들어 진다. 
> 
> **한 Layer에서 원하는만큼 여러 개의 Filter를 사용**할 수 있고, 아래 예시와 같이 5 X 5 Filter가 6개 있다면 총 6개의 Activation Map을 얻게 되는 것이다. 
> 
> <img width="696" alt="스크린샷 2024-02-02 오후 9 43 20" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/d9d410f0-efb1-47c8-8c48-959a68016b77">
> 
> <img width="659" alt="스크린샷 2024-02-02 오후 9 41 49" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/3a96878b-99ec-4dfe-a0c9-73d81fb9916f">
> 
> CNN은 다음과 같은 방식으로 Convolution Layer의 연속적인 형태를 갖는다. 
> 
> 각각 쌓아 올리게 되면 Linear Layer로 된 Neural Network가 되고, 그 사이사이에 ReLU와 같은 Activation Function을 넣는 것이다. 
> 
> 그렇게 되면 **Conv-ReLU는 반복, 가끔 Pooling Layer도 들어가고 마지막 끝단에 FC-Layer**를 가진다. 여기서 Pooling은 Activation Map의 크기를 줄이는 역할을 한다. 
> 
> 각 Layer의 출력은 다음 Layer의 입력이 되고, 각 Layer는 여러 개의 Filter를 가지고 각 Filter마다 출력 Map을 만든다. 따라서 여러 개의 Layer들이 쌓인 뒤 보면 각 Filter들이 계층적으로 학습하는 것을 볼 수 있다. 
> 
> <img width="669" alt="스크린샷 2024-02-02 오후 9 53 00" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/5514ec14-963e-4ff1-8ad8-5ad97996f235">
> 
> 앞쪽의 Filter들은 Low-Level Feature를 학습한다. → Edge 끝부분 & 가장자리 
> 
> Mid-Level을 보면 좀 더 복잡한 특징을 가지게 된다. → Cornner and Blobs 모서리 & 작은 색 부분 
> 
> High-Level Features를 보면 좀 더 객체와 닮은 것들이 출력으로 나온다. 
> 
> <img width="636" alt="스크린샷 2024-02-02 오후 9 56 22" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/3edcb9f0-0847-4d7c-b1bb-aaa6fe389607">
> 
> **Level이 높아질 수록 더 많은 특징을 담고, Layer의 계층에 따라 단순 또는 복잡한 특징이 존재한다.** 
> 

### **Spatial Dimension**

> **32 X 32 X 3 이미지가 있고, 이 이미지를 5 X 5 X 3 Filter를 가지고 연산을 수행할 때, 28 X 28 Activation Map이 생기는 이유**
> 
> 
> 7 X 7 입력에 3 X 3 Filter가 있을 때 이 Filter를 가장 왼쪽 상단에 위치시키고, 해당 값들의 내적을 수행한다. 
> 
> 그 다음 Filter를 오른쪽으로 한칸 움직이는데 이 때 또 하나의 값을 얻을 수 있다. 계속 반복하다 보면 5 X 5 출력을 얻게 된다. 
> 
> 아래 예시를 보면 슬라이드 Filter는 좌우 방향으로 5번만, 상하 방향으로 5번만 수행 가능하기 때문이다. 
> 
> Filter 슬라이딩을 한칸씩 진행했는데, 이때 **움직이는 칸을 Stride**라고 한다. 
> 
> 아래의 경우 Stride = 1을 사용했다. 
> 
> <img width="606" alt="스크린샷 2024-02-03 오전 12 16 13" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/d6e27b72-7312-49f8-8399-b9913238a9bf">
> 
> Stride = 2인 경우, 다시 왼쪽 상단부터 시작해 움직이며 1칸은 건너뛰고 그 다음 칸으로 이동해서 계산한다. 이 경우 출력은 3 X 3이 된다. 
> 
> <img width="606" alt="스크린샷 2024-02-03 오전 12 20 09" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/c01d0619-7dc6-49b4-94dd-5f7355f50a71">
> 
> <img width="606" alt="스크린샷 2024-02-03 오전 12 20 49" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/eee331b8-7142-4108-9341-6e0ec18f0eb3">
> 
> Stride = 3인 경우, 이미지를 슬라이딩해도 Filter가 모든 이미지를 커버할 수 없다. 
> 
> 이미지에 잘 맞아떨어지지 않기 때문에 실제로 아래 경우 잘 동작하지 않는다고 볼 수 있다. 이로 인해 불균형한 결과를 가져올 수 있기 때문에 Stride 값을 아래와 같이 설정하면 안된다. 
> 
> 이 상황을 방지하기 위해 **출력의 크기가 어떻게 될 것인지 계산하는 수식**을 사용할 수 있다. 입력의 차원이 N이고 Filter 크기는 F, Stride 값이 주어지게 되면 출력의 크기는 **(N - F) / Stride + 1**이다. 
> 
> 이 수식을 이용하면 어떤 Filter 크기를 사용해야 하는지 알 수 있다. 
> 
> <img width="700" alt="스크린샷 2024-02-03 오전 12 25 49" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/1e946b9d-aa27-4506-bdd9-56806babe347">
> 

> **Zero-Padding**은 가장자리 Filter 연산을 수행하도록 하고 출력의 크기를 의도대로 만들어준다. 이를 이용해 **입력 크기를 유지할 수 있고 코너를 처리**할 수 있다.
> 
> 
> Zero-Padding은 이미지의 가장자리에 0을 채워 넣음으로써 왼쪽 상단의 자리에서도 Filter 연산을 수행할 수 있게 된다. 
> 
> Zero-Padding을 하면 새 출력이 7이 되어 **출력의 차원이 입력의 차원과 같아진다.** → 출력은 7 X 7 X Filter의 갯수
> 
> <img width="643" alt="스크린샷 2024-02-04 오후 4 58 04" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/1ffd0fd8-33d0-420f-bd3a-517b5c99166a">
> 
> **Zero-Padding 하지 않았을 때, Layer가 쌓이게 되면 크기는 급속도로 줄어들게 될 것이고 Activation Map이 작아지게 된다.** 
> 
> Zero-Padding은 Filter가 닿지 않는 모서리 부분에서도 값을 뽑을 수 있게 해주는 하나의 방법이다. Zero가 아닌 Mirror나 Extend 하는 방법도 있다. 
> 
> 아래 예시는 Conv Layer의 요약본으로 몇 개의 Filter를 쓸 것인지, Filter의 크기는 몇인지 Stride 값과 Zero-Padding의 값을 정해야 한다. 그리고 출력의 크기가 어떻게 될 것인지 또한 계산해야 한다. 
> 
> 일반적으로 Filter 크기는 3 X 3, 5 X 5이고 Stride는 1이나 2, Padding은 설정에 따라 조금씩 다르게 하고 Filter의 갯수는 32, 64, 128, 512 등 2의 제곱수로 한다. 
> 
> <img width="596" alt="스크린샷 2024-02-04 오후 5 00 56" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/e8fd2b43-e5df-4722-8ef8-d9bcd0841ff0">
> 
> 5 X 5 Filter가 있다면 한 뉴런의 Receptive Filed가 5 X 5라고 할 수 있다. Receptive Filed는 한 뉴런이 한 번에 수용할 수 있는 영역을 의미하고 이는 Filter의 크기와 같다. 
> 
> <img width="647" alt="스크린샷 2024-02-04 오후 5 34 15" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/45b2621b-e81a-4341-88c2-26cef6bd9bfc">
> 
> Conv Layer의 출력이 작을수록, 그리고 Stride 값이 커질수록 FC Layer에서 필요한 파라미터의 수가 작아질 것이고 이는 **Overfitting**과 관련이 있다.
> 
> <img width="596" alt="스크린샷 2024-02-04 오후 5 30 47" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/f0d1e359-bde5-4b1d-ae20-35d0c0b6e010">
> 
> CNN에는 **Conv Layer**와 **Pooling Layer**, 그리고 다른 비선형 연산들도 있다. 
> 
> Pooling Layer는 **Down Sampling을 하는 것**으로 Representation들을 더 작고 관리하기 쉽게 만들어준다. 
> 
> 위 예시를 보면 224 X 224 X 64인 입력이 있을 때, 이를 112 X 112 X 64로 공간을 줄여준다. 이때 Depth에는 영향을 주지 않는다. 
> 
> <img width="610" alt="스크린샷 2024-02-04 오후 5 41 24" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/ab85066d-2438-44e1-b0e6-c736ac0ef748">
> 
> 일반적으로 **Max Pooling**이 사용되는데, Pooling에도 Filter 크기를 정할 수 있다. 
> 
> 2 X 2 Filter가 있고 Stride가 2일 때 Conv Layer가 했던 것처럼 슬라이딩 하면서 연산을 수행하지만, 내적을 하는 것이 아니라 Filter 안에 가장 큰 값 중에 하나를 고른다. 
> 
> 위 사진을 보면 빨간색 영역에서는 6이 가장 크고, 초록색 영역에서는 8이 제일 크다. 나머지도 마찬가지로 3과 4의 값이 나온다. 
> 
> Pooling 할 때는 보통 겹치지 않는 것이 일반적이다. 기본적으로 Down Sample을 하고 싶은 것이기 때문에 한 지역을 선택하고 값 하나를 뽑고, 또 다른 지역을 선택하고 값 하나를 뽑는 식으로 진행된다. 
> 
> Max Pooling 대신 Average Pooling도 사용할 수 있지만, Max Pooling은 그 지역이 어디든 어떤 신호에 대해 얼마나 그 Filter가 활성화 되었는지를 알려주기 때문에 사용한다. 
> 
> <img width="581" alt="스크린샷 2024-02-04 오후 5 45 28" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/a1c6c91f-1c83-4b40-a0f7-15f140e9d8ef">
> 
> Pooling Layer에는 몇 가지 선택지가 있는데, 입력이 W, H, D이면 이를 통해 Filter 크기를 정해줄 수 있다. 
> 
> 여기에 Stride까지 정하면, 앞서 Conv Layer에서 사용했던 수식을 그대로 이용해 출력 값을 계산할 수 있다. → **((W - Filter Size) / Stride + 1)**
> 
> 보통 Pooling Layer에서는 Zero Padding을 사용하지 않는다. Pooling 할 때 Padding을 고려하지 않고 그냥 Down Sample만 하면 된다. 
> 
> 가장 널리 사용되는 Filter 크기는 2 X 2, 3 X 3이고 보통 Stride는 2로 한다. 
> 
> <img width="627" alt="스크린샷 2024-02-04 오후 5 49 53" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/15affd0c-9429-488b-b8ac-e4afa6437d39">
> 
> FC Layer는 마지막에 존재한다. 마지막 Conv Layer의 출력은 3차원 Volume으로 이루어진다. 
> 
> 이 값들을 전부 펴서 1차원 Vector로 만들고 FC Layer의 입력으로 사용한다. 
> 
> 이렇게 되면 Conv Net의 모든 출력을 서로 연결하는 것이다. 마지막 Layer부터는 공간적 구조를 신경쓰지 않아도 된다. 
> 
> 전부 하나로 통합시키고 최종적인 추론을 하면 Score가 출력된다. 