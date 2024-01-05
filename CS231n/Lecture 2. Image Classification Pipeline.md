## **Lecture 2.  Image Classification Pipeline**

### **Image Classification**

> **Image Classification**이란 **입력 이미지를 시스템에서 미리 정해진 라벨, 분류된 이미지 집합 중 어디에 속하는지 판단하는 것**을 말한다.
> 
> 
> 이때 **이미지란 0-255 정수 범위 값을 가지는 너비(W) X 높이(H) X 채널(C)의 크기를 가지는 3차원 배열을 말한다.** 
> 
> <img width="539" alt="스크린샷 2024-01-05 오후 8 50 00" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/17eeaaa1-ab4f-4cb6-ad4b-6e516262f732">
> 
> 3은 RGB로 구성된 3개의 채널을 의미한다. 
> 
> 시스템이 고양이 이미지를 입력받으면 RGB 값을 기준으로 격자 모양의 숫자를 나열하여 인식한다. 즉, 고양이 이미지가 아주 큰 격자 모양의 숫자 집합으로 보이는 것이다. 
> 
> **각 픽셀은 RGB 3개의 숫자로 표현되고 단순히 거대한 숫자 배열로 보이기 때문에 시스템이 이를 고양이로 인식하는 것은 매우 어려운 일**이다. 
> 
> 고양이라는 라벨은 의미상의 라벨일 뿐, 픽셀 값과는 큰 차이가 있다. 
> 
> 이는 매우 어려운 과제인데, 카메라 위치, 조명의 변화, 혹은 객체 자체의 변화 등의 상황이 주어졌을 때, 시스템이 바라보는 이미지의 픽셀은 전부 바뀌지만 이미지가 고양이를 나타낸다는 사실에는 변함이 없기 때문이다. 
> 

> **Challenges**
> 
> 1. **Viewpoint Variation, 시점 변화**
>     
> <img width="600" alt="스크린샷 2024-01-05 오후 9 11 32" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/f0c80be8-3a56-4bfe-8f51-d684f84ae1f8">
>     
> 2. **Background Clutter**
>     
> <img width="517" alt="스크린샷 2024-01-05 오후 9 13 04" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/d8a9e8f4-6f9d-46cb-9058-101713d98199">
>     
> 3. **Illumination, 조명 상태**
>     
> <img width="640" alt="스크린샷 2024-01-05 오후 9 11 56" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/e394587e-9c25-495c-a7bc-588a0bac57b4">

>     
> 4. **Occlusion, 폐색, 가려짐**
>     
> <img width="592" alt="스크린샷 2024-01-05 오후 9 12 30" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/a0a46a82-e67f-4eba-b4ac-37428f1817e5">
>     
> 5. **Deformation, 변형**
>     
> <img width="652" alt="스크린샷 2024-01-05 오후 9 12 14" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/1292a726-0aaa-4284-871a-1f082ba0d0bc">
>     
> 6. **Intraclass Variation, 내부 클래스의 다양성** 
>     
> <img width="408" alt="스크린샷 2024-01-05 오후 9 13 44" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/11e7c5a0-3986-4082-a946-d90494382dcf">
>     

> 이를 해결하기 위해 시도해볼 수 있는 방법으로, Gradient에 의한 특정 패턴을 정의하는 방법이 있다. 이 방법은 Deformation에서 큰 이점을 가지지만 Location 정보가 부족하다는 단점이 있다.
> 
> <img width="635" alt="스크린샷 2024-01-05 오후 9 19 48" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/de50d9cc-0c7e-45a6-ac49-25a5b2ed1264">
> 
> 따라서 **좋은 이미지 분류 모델은 클래스 간 변동에 대한 민감도를 유지하며, 모든 변동의 교차곱에 대해 불변**해야 한다. 
> 

> 이미지 분류를 함수화 한다면 다음과 같다. **이미지를 입력받고, 클래스 라벨을 반환**한다.
>
> <img width="518" alt="스크린샷 2024-01-05 오후 9 16 17" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/02b231e4-4c83-42b3-b4c0-d5d5a7aaca55">
> 

> **Data Driven Approach, 데이터 중심 접근 방법**
> 
> 
> 객체의 특징을 규정하지 않고 다양한 사진과 라벨을 수집해 기계 학습 분류기로 학습하여 사진을 새롭게 분류하는 방식이다. 
> 
> <img width="615" alt="스크린샷 2024-01-05 오후 9 23 54" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/0fc286e8-6465-427c-94b0-19ce5a0931d3">
> 

### **Nearest Neighbor, NN**

> **Nearest Neighbor, 최근접 이웃법**은 입력 받은 데이터를 저장한 다음 새로운 입력 데이터가 들어오면, **기존 데이터와 비교해 가장 유사한 데이터를 찾는 방식**이다.
> 
> 
> 아주 단순한 분류기로, 학습 단계에서 모든 학습 데이터를 기억해 모델을 만든 후, 예측 단계에서 새로운 이미지가 들어왔을 때 기존의 학습 데이터를 비교해 가장 유사한 이미지로 이미지의 라벨을 예측한다.
> 
> 단순하지만 **Data Driven Approach**로는 좋은 알고리즘이다. 
>
> <img width="639" alt="스크린샷 2024-01-05 오후 9 40 49" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/ddc31cf7-0434-4d88-a444-6ee9f9634b03">
> 
> 이미지를 비교할 때 **L1 Distance**라는 함수를 사용한다. 
> 
> 이 함수는 이미지의 각 픽셀을 비교하는데, **Test, Train 이미지의 같은 자리에 픽셀을 서로 빼고 절댓값을 취하는 방식**으로 두 이미지의 픽셀차를 계산하고 모든 차이값을 더한다. 
> 
> 이 방법의 단점은 모든 사진의 픽셀 값을 계산하기 때문에 예측 과정에서 소요되는 시간이 상당하다는 것이다. 즉, **Train은 O(1), Test는 O(N)의 시간 복잡도**를 가진다. 
>
> <img width="499" alt="스크린샷 2024-01-05 오후 9 54 58" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/05948612-8253-4a22-8fb0-731b82ddc4dd">
> 
> 위 사진은 Nearest Neighbor을 적용했을 때 모습으로 2차원 평면 상의 각 점은 학습 데이터이고, 점의 색은 라벨을 나타낸다. 
> 
> 초록색 점 사이 노란색 점이 끼어있는 경우, 초록색 영역이 파란색 영역을 침범하고 있는 경우에 가까운 이웃만 확인하는 NN 알고리즘 특성상 문제가 발생할 수 있다. 
> 
> 이러한 비효율을 개선한 방법이 **K-Nearest Neighbor**이다. 
> 
> 가까운 이웃만 찾는 것이 아닌 **Distance Metric을 사용해 가까운 이웃을 K개만큼 찾고 이웃간에 투표, 득표수가 가장 높은 라벨로 예측하는 방법**이다. 
>
> <img width="648" alt="스크린샷 2024-01-05 오후 10 00 14" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/e8be9b4b-e9d2-4432-952f-335ea9cb6114">
> 
> 예측값을 보았을 때, KNN도 성능이 좋지는 않다. K값을 높이거나 서로 다른 점들을 어떻게 비교할 것인지 결정함으로서 모델의 성능을 향상시킬 수는 있다. 
> 
> 기존 모델의 경우 L1 Distance를 사용했는데, 제곱 합의 제곱근을 거리로 사용하는 방법인 Edclidean Distance를 사용할 수도 있다. 
> 
> **만일 입력값의 요소들이 개별적인 의미를 가지고 있다면 L1이 더 잘 어울리지만, 요소들간의 실질적인 의미를 모른다면 L2 Distance가 더 잘 어울린다.** 
>
> <img width="577" alt="스크린샷 2024-01-05 오후 10 03 13" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/79376b98-1100-440f-ac06-ee2c0e8d34d3">
> 
> <img width="569" alt="스크린샷 2024-01-05 오후 10 03 28" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/4f4539c5-ced9-4746-b31a-852b3c78132c">
> 
> K-Nearest Neighbor을 사용하려면 학습 전 사전에 **K**와 거리 척도인 **Hyper-Parameter**를 선택해야 한다. 
> 
> Hyper-Parameter를 선택하는 방법은 Problem-Dependent이다. 
> 
> 단순히 학습 데이터의 정확도와 성능을 최대화하는 Hyper-Parameter를 선택하는 것은 좋지 않다. 학습 데이터에는 없던 테스트용 데이터를 넣었을 때 아주 안좋은 성능이 나올 수 있기 때문이다. 
> 
> 전체 데이터를 쪼개 일부를 테스트용으로 사용하고, 학습시킨 모델들 중 가장 잘 맞는 모델을 선택하는 방법도 있는데, 이는 테스트용 데이터에서만 잘 동작하는 Hyper-Parameter를 고른 것일 수 있기 때문에 이 또한 좋지 않다. 
> 
> <img width="614" alt="스크린샷 2024-01-05 오후 10 25 14" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/71a63772-6e03-4d47-b464-6a9af185c70c">
> 
> 중요한 것은 한번도 보지 못한 데이터를 얼마나 잘 예측하는가이다. 
> 
> **일반적인 방법으로는 데이터를 3개로 나누는 것**이다. ****데이터의 대부분을 Train-Set, 일부는 Validation-Set, 나머지는 Test-Set로 나눈다. 
> 
> **다양한 Hyper-Parameter로 Train-Set을 학습 후 Validation-Set으로 검증을 거치고 가장 좋았던 모델을 사용해 Test를 수행**한다. 
> 
> 또 다른 Hyper-Parameter 선택 전략은 **Cross Validation, 교차 검증**으로 우선 Test-Set을 정해두고 나머지 데이터는 Train-Set, Validation-Set으로 나누는데 데이터를 여러 부분으로 나누고, **번갈아가면서 Validation-Set을 지정**해주는 방법이다. 
> 
> <img width="576" alt="스크린샷 2024-01-05 오후 10 25 31" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/bfc27b67-3d41-43db-a31c-5e8a354cdde7">
> 
> 실제로 입력값이 이미지인 경우, K-NN은 잘 사용하지 않는다. 
> 
> 구현이 단순하고 쉬우며, 학습 시간이 전혀 소요되지 않지만 테스트용 데이터 확인 시 모든 학습 데이터와 비교해야 하기 때문에 계산량이 많아지므로 매우 느리다. 
> 
> 또한 L1, L2 Distance가 이미지 사이 거리를 측정하기에는 적절하지 않고, Curse of Dimensionality 문제도 있다. 
> 
> <img width="595" alt="스크린샷 2024-01-05 오후 10 32 59" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/c638f56c-bfa9-47e2-a5bd-0c2f3c79e7cc">
> 
> <img width="618" alt="스크린샷 2024-01-05 오후 10 33 29" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/6b0ed1dd-8fd6-48e9-8b10-b0616aea9e76">
> 

### Linear Classification, 선형 분류

> **Linear Classification**은 Parametric 모델의 단순한 형태이다. **Neural Network와 Convolution Neural Network의 기반 알고리즘**이다.
> 
> 
> 이 모델에서는 입력 이미지를 보통 **X**로 표기하고, Parameter 혹은 가중치 **W**라고 하는 두 가지 요소가 있다. 
> 
> W * X에 **Bias**를 더하는데 이는 **편향값**으로 입력과 직접적인 관계를 가지지 않으나 **이미지 라벨의 불균형한 상태를 보완하기 위해 사용**된다. 
> 
> <img width="625" alt="스크린샷 2024-01-05 오후 10 45 35" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/f21ed56f-7903-4e13-a85c-08079faf9e02">
> 
> <img width="646" alt="스크린샷 2024-01-05 오후 10 48 20" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/a8e05b81-1aee-4f4f-8d26-4cd0248dd3b2">
> 
> <img width="617" alt="스크린샷 2024-01-05 오후 11 03 50" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/ddb8ecef-8124-4f8f-b6f0-a500bd6a307b">
> 
> 위 사진을 예로 들면, 이미지 X와 가중치 W를 곱해 카테고리 스코어 값(f(x, w))을 10개 만든다. 이때 고양이 카테고리 스코어가 가장 높다면 입력 이미지 X가 고양이일 확률이 크다는 것을 의미한다. 
> 
> 이미지를 고차원 공간으로 보게 되면, Linear Classification는 각 클래스를 구분해주는 선형 경계 역할을 하지만 일차 함수 직선으로 분류되지 않는, 즉 데이터의 분포에 따라 선형으로 분류할 수 없는 데이터가 대부분이다. 
> 
> <img width="630" alt="스크린샷 2024-01-05 오후 11 04 51" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/73006515-071a-44f3-b649-f6532d37e841">
> 
> 1. 사분면에서 반대 위치에 Decision Boundaries, 결정 경계가 있는 경우
> 2. 원형의 Decision Boundaries, 결정 경계가 있는 경우
> 3. 3개 이상의 독립적인 Decision Boundaries, 결정 경계가 있는 경우