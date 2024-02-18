## **Lecture 4. Back Propagation and Neural Networks**

### Back Propagation

> Gradient를 계산할 때, Numerical Gradient로 초기 계산을 하고, 실제 연산은 Analytic Gradient 방법을 사용해 최종 파라미터를 확인했다.
> 
> 
> 이러한 연산 과정을 정리한 것을 **Computational Graph**라고 한다. 
> 
> <img width="633" alt="스크린샷 2024-01-26 오후 7 44 56" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/75c6c6c3-58ac-4223-b1c6-3f09c84573db">
> 
> Vector x와 가중치 연산을 통해 s를 계산한 후 Loss Function을 통해 Loss 값을 최소화 할 수 있도록 조정하는데, 이때 Overfitting을 방지하기 위해 정규화를 사용한다. 
> 

> **Front Propagation**
> 
> 
> <img width="699" alt="스크린샷 2024-01-26 오후 7 50 51" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/70d58b79-e24a-4423-b871-940b8e1940fe">
> 
> x, y, z 값이 주어지고 왼쪽에서 오른쪽 노드로 건너가며 연산이 진행될 시, 최종 f값이 -12로 계산된다. 
> 

> **Back Propagation**
> 
> 
> x, y, z가 f(x, y, z)에 어떤 연산을 미치는지 알아보기 위해 말 그대로 **거꾸로 연산하는 방법**이다. 
> 
> 각각 x, y, z가 output인 f에 미치는 영향을 생각했을 때 편미분으로 표현하게 되면 df / dx, df / dy, df/ dz로 표현할 수 있다. 
> 
> q = x + y이고, f = qz로 주어진다. 
> 
> 덧셈 연산과 곱셈 연산에 대해 미분 시 미분 값은 1이고, 곱셈 연산에서는 서로의 값을 가진다. 
> 
> 최종 f에 영향을 미치는 정도를 파악하기 위해 Gradient 계산을 하면 값은 1이다. 
> 
> z가 f에 영향을 미치는 정도를 생각했을 때 f = qz, 곱셈의 형태이다. 
> 
> 즉 q 값은 3이다. 
> 
> q가 f에 영향을 미치는 것을 보면 당연히 z값인 -4가 되는 것 같지만 아니다. 
> 
> q와 z는 y와 직접적으로 연결되어 있는데 x, y는 그렇지 않다. 이 때 **Chain Rule**을 사용한다. 
> 
> x → q → f 순서이므로 (x가 q에 미치는 영향) * (q가 f에 미치는 영향)을 계산해주면 된다. 
> 
> x + y의 미분 값은 1이고 이 값을 Local Gradient라고 부른다. 
> 
> y와 x가 q에 미치는 영향은 값이 1이다. 따라서 1 * q의 값으로 계산하면 -4가 된다. 
> 
> <img width="699" alt="스크린샷 2024-01-26 오후 7 58 42" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/6af6e806-71df-4dd1-be1e-a218cb364f0e">


> **Jacobian Matrix**
>
> 
> Vector 함수 Jacobian Matrix는 해당 함수의 편도 함수 행렬로 **Local Gradient * Global Gradient 연산을 할 때 필요한 행렬**이다. 
> 
> <img width="699" alt="스크린샷 2024-01-26 오후 8 16 21" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/a65b4d33-2a4e-48fb-a208-01868a9d61c9">
> 
> <img width="850" alt="스크린샷 2024-01-26 오후 8 40 15" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/00176353-b50e-4b16-8501-a11415a94ec6">
> 
> Jacobian Matrix 크기가 4096 * 4096으로 주어지면 minibatch를 이용한 연산 시, 실용적이지 못하다. 따라서 연산을 하지 않고, 출력에 대한 x의 영향을 구할 때 일부는 0으로 채운다. 
> 

### Neural Network

> **Neural Network**은 인공 뉴런이나 노드로 구성된 인공 신경망을 말한다.
> 
> 
> <img width="695" alt="스크린샷 2024-01-26 오후 8 32 52" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/ecfb452a-4248-49c2-b35c-a1478791adc6">
> 
> 기존 Linear Layer 행렬 곱에 의한 스코어 함수 f = Wx에서 가중치 행렬 W의 각 행이 클래스 각각의 탬플릿과 비슷한지 비교하였다. 
> 
> 각 클래스마다 오직 하나의 탬플릿만을 가지고 있는데, 다중 레이어 신경망은 다양하게 이미지를 분류할 수 있도록 해준다. 
> 

> **Activation Functions**
> 
> 
> 여러 겹층의 선형 분류기로 구성된 Neural Network Activation Function을 사용해 0-1 사이의 값으로 분류, Score를 표현한다. 
> 
> <img width="695" alt="스크린샷 2024-01-26 오후 8 22 03" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/22bd0992-942f-4b5c-b732-3c404b89156c">
> 
> 위 계산 과정에 bias 값을 추가해주는 이유는 파라미터의 선호도를 조절해주기 위함이다. 
> 
> **Activation Function**이란 **입력된 데이터의 가중 합을 출력 신호로 변환하는 함수**를 말한다. 
> 
> <img width="695" alt="스크린샷 2024-01-26 오후 8 36 32" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/b272e914-2422-4fc5-b7d4-01310beb1363">
