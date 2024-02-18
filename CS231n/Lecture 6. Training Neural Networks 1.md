## **Lecture 6. Training Neural Networks 1**

### **Activation Function**

> CNN은 Spatial Structure를 보존하기 위해 CNN Layers를 사용하는 NN의 한 종류이다.
> 
> 
> Conv Filter가 입력 이미지를 슬라이딩해서 계산한 값들이 모여 각 출력 Activation Map을 만든다. 
> 
> Conv Layer는 각 Layer마다 다수의 Filter를 사용할 수 있고, 각 Filter는 서로 다른 Activation Map을 생성한다. 
> 
> 가중치 또는 파라미터 값을 알아내야 하는 것이고, Optimization을 통해 네트워크의 파라미터를 학습할 수 있다. 
> 
> 파라미터 업데이트를 하며 Loss가 줄어드는 방향으로 가는 것이 목적이며, 이를 위해서는 Gradient의 반대 방향으로 가면 된다. 
> 
> Mini-Batch Stochastic Gradient Descent는 우선 데이터의 일부만 가지고, Forword Pass를 수행한 뒤 Loss를 계산한다. 
> 
> 그리고 Gradient를 계산하기 위해서는 Backprop을 수행한다. 
> 

> Layer로 데이터 입력이 들어오면 FC와 CNN에 가중치와 곱, 즉 **비선형 연산**을 거친다.
> 
> 
> <img width="403" alt="스크린샷 2024-02-18 오후 12 36 42" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/8bdb4b02-c7ea-486c-b475-9c177fd9a260">
> 
> <img width="458" alt="스크린샷 2024-02-18 오후 12 43 07" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/5745dad9-99d0-498b-9cbc-5cb5b91770fe">
> 

> **Sigmoid**
> 
> 
> Sigmoid는 **출력이 (0, 1) 사이의 값이 나오도록 하는 선형 함수**이다. 
> 
> <img width="458" alt="스크린샷 2024-02-18 오후 12 46 32" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/11f20429-0c01-416a-8e9a-df9682da79bf">
> 
> **Sigmoid 단점**
> 
> 1. Saturated Neurons가 Gradient 값을 0으로 만든다. 
> 2. 원점 중심이 아니다. 
> 3. 지수 함수가 계산량이 많다. 
> 
> <img width="470" alt="스크린샷 2024-02-18 오후 12 56 58" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/b921a07c-7177-4e62-9653-7a8dcee65e73">
> 
> Saturate는 포화로 입력이 너무 작거나 클 경우 값이 변하지 않고, 일정하게 1로 수렴하거나 0으로 수혐하는 것을 포화라고 생각하고 Gradient의 값이 0인 부분을 의미한다. 
> 
> **Gradient의 값이 0이 되는 것의 문제점**은 Chain Rule 과정을 생각했을 때, **Global Gradient 값이 0이되면** 즉, 결과 값이 0이 되면 **Local Gradient 값도 0이 된다.** 
> 
> 따라서 입력에 있는 Gradient 값을 구할 수 없다. 
> 
> **원점의 중심이 아닌 것**은 **출력의 값이 항상 양수일 때 다음 입력으로 들어갔을 때에도 항상 양수**이게 된다. 
> 
> <img width="465" alt="스크린샷 2024-02-18 오후 12 53 53" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/b5e2e5fa-8d57-47ac-8133-a3c8b69039f4">
> 
> 다음 Layer에서 W의 값을 업데이트 할 때 항상 같은 방향으로 업데이트가 된다. 
> 
> 원하는 Vector의 방향이 파란색일 때, W와 같은 경우 제 1사분면과 제 3사분면으로 업데이트가 되기 때문에 원하는 방향으로 업데이트를 하기 힘들다. 
> 

> Sigmoid의 단점을 보안하기 위해 나온 함수가 **tanh(x)** 이다.
> 
> 
> <img width="465" alt="스크린샷 2024-02-18 오후 12 56 12" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/b57b9c3f-7494-463a-ac55-e3e957b5b91a">
> 
> tanh(x)는 Sigmoid와 유사하지만 **범위가 (-1, 1)**이다. 
> 
> 차이점은 **Zero-Centered**라는 것이다. Zero-Centered이기 때문에 Sigmoid의 문제가 해결된다. 
> 
> 하지만 Saturation 때문에 **여전히 Gradient가 죽는다.** 위 사진의 tanh(x) 그래프를 보면 여전히 Gradient가 평평해지는 구간이 있다. 
> 

> **ReLU, Rectified Linear Unit**
> 
> 
> ReLU 함수는 **f(x) = max(0, x)** 이다. Element-Wise 연산을 수행하며, 입력이 음수면 값이 0이 된다. 그리고 양수면 입력 값 그대로 출력한다. 
> 
> ReLU는 자주 쓰이는데, 기존 Sigmoid와 tanh(x)한테 있었던 문제점을 가지고 비교해보면, **ReLU은 양수 값에서 Saturation 되지 않는다.** 
> 
> **입력의 절반이 Saturation 되지 않는다**는 점이 큰 장점이다. 
> 
> 그리고 Sigmoid 함수 안에는 지수 항이 있지만, **ReLU는 단순히 Max 연산이라 계산이 매우 빠르다. 즉, 효율이 뛰어나다.** 
> 
> 실제로 Sigmoid나 tanh(x)보다 수렴 속도가 거의 6배 정도 빠르다. 
> 
> <img width="470" alt="스크린샷 2024-02-18 오후 1 14 17" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/74774d72-3f83-4467-bd31-4d7de869d9ad">
> 
> <img width="470" alt="스크린샷 2024-02-18 오후 1 15 03" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/200d197a-dcac-4d3f-b467-d230a36ac5c3">

> 
> ReLU에도 **단점**은 존재한다. 
> 
> ReLU는 **Zero-Centered가 아니다.** tanh(x)를 사용하면서 이 문제가 해결 되었는데, ReLU가 다시 이 문제를 가지고 있다. 
> 
> 또한 양수에서는 Saturation 되지 않지만, **(-)의 값은 0으로 만들어 버리기 때문에 Data의 절반만 Activate**하게 만든다는 것이다. 
> 
> <img width="470" alt="스크린샷 2024-02-18 오후 1 14 50" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/9d7dec1e-fed0-49ac-b05b-97123fe023c7">
> 

> **Leaky ReLU**
> 
> 
> <img width="470" alt="스크린샷 2024-02-18 오후 1 38 58" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/5aa6873c-6a5c-4267-a4a5-02e88b15c909">
> 
> 
> ReLU와 유사하지만 **Negative Regime에서 0이 아니다.** 이 함수는 Negative에도 기울기를 살짝 주게 되면 앞에 함수들에게서 있었던 문제의 상당 부분이 해결된다. 
> 
> **Leaky ReLU의 경우 Negative Space에서도 Saturation 되지 않는다.** 
> 
> 그리고 여전히 계산이 효율적이라 Sigmoid나 tanh(x)보다 수렴을 빨리 할 수 있다. 
> 
> **Dead ReLU 현상도 없다.** 
> 

> **PReLU, Parametric Rectifier**
> 
> 
> <img width="470" alt="스크린샷 2024-02-18 오후 1 36 51" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/630b1f47-5ced-4408-a3ef-10f0a5ca00d6">
> 
> **PReLU는 Negative Space에 기울기가 있다는 점**에서 **Leaky ReLU와 유사**하다. 
> 
> 기울기가 Alpha라는 파라미터로 결정되는데, Alpha를 딱 정해놓은 것이 아니라 Backprop으로 학습시키는 파라미터로 만든다. 
> 
> **활성 함수가 조금 더 유연해질 수 있다.** 
> 

> **ELU**
> 
> 
> <img width="470" alt="스크린샷 2024-02-18 오후 1 37 27" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/e77b9ffd-a0ba-4bf3-9f7b-6216ee2e6e45">
> 
> ELU는 ReLU의 장점을 가져오지만, **Zero-Mean에 가까운 출력값**을 보인다. 
> 
> Zero-Mean에 가까운 출력은 Leaky ReLU, PReLU가 가진 장점이다. 
> 
> Leaky ReLU와 비교해보면, **ELU는 Negative에서 기울기를 가지는 것 대신에 또 다시 Saturation 된다.** 
> 
> ELU Function이 주장하는 것은 Saturation이 Noise에 강할 수 있다는 것이다.  
> 
> ELU는 ReLU와 Leaky ReLU의 중간 정도로 Leaky ReLU처럼 Zero-Mean의 출력을 내지만 Saturation의 관점에서 ReLU의 특성도 가지고 있다. 
> 

> **Maxout Neuron**
> 
> 
> <img width="470" alt="스크린샷 2024-02-18 오후 1 51 16" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/7162c961-ff02-4c67-a9db-ee9dd207a6ff">
> 
> 이전까지의 Activation Function과 다르게 입력을 받아들이는 특정 형식을 미리 정의하지 않는다. 
> 
> 대신 w1에 x를 내적한 값 + b1과 w2에 x를 내적한 값 + b2의 최대값을 사용한다. 
> 
> Maxout은 이 두 함수 중 최대값을 취한다. ReLU와 Leaky ReLU의 좀 더 일반화 된 형태인데, **Maxout은 두 개의 선형 함수를 취하기 때문이다.** 
> 
> Maxout 또한 선형이기 때문에 Saturation 되지 않으며 Gradient가 죽지 않는다. 
> 
> 문제는 뉴런 당 파라미터의 수가 두 배가 되어 w1과 w2를 지니고 있어야 된다는 점이다. 
> 

### **Data Pre-Processing**

> 데이터 전처리 과정에서는 주로 **Zero-Centered, Normalized, PCA, Whitening**과 같은 처리를 한다.
> 
> 
> Zero-Centered나 Normalized를 하는 이유는 모든 차원이 동일한 범위에 있어 전부 동등한 기여를 할 수 있도록 하는 것이다. 
> 
> PCA나 Whitening은 더 낮은 차원으로 Projection하는 것인데, 이미지 처리에서 이런 전처리 과정을 거지치는 않는다. 
> 
> 기본적으로 이미지는 Zero-Centerd 과정만 거친다. 실제 모델에서는 Train Data에서 계산한 평균을 Test Data에도 동일하게 적용한다. 
> 
> <img width="470" alt="스크린샷 2024-02-18 오후 1 57 45" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/a1eba214-ea22-4d62-9cbc-716d4e510205">
> 
> <img width="470" alt="스크린샷 2024-02-18 오후 1 58 15" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/98b36b2e-4fe6-4e42-b86f-026ffdbf65b6">
> 

### **Weight Initialization**

> 초기값을 얼마로 잡아야 최적의 모델을 구할 수 있을까 ?
> 
> 
> 만일 초기값을 0으로 한다면, 모든 뉴런은 동일하게 일을 하게 될 것이다. 
> 
> 즉, 모든 Gradient 값이 같게 될 것이고, 이것은 아무 의미가 없다. 
> 
> <img width="477" alt="스크린샷 2024-02-18 오후 2 01 00" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/b76739f6-9e73-4fde-85b1-e04687f74d57">
> 
> 만약 표준 편차를 키우게 되면 어떻게 될까 ?
> 
> <img width="475" alt="스크린샷 2024-02-18 오후 2 02 16" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/8c17f20f-4eb9-4dbd-8292-9e2bfacac766">
> 
> 가중치가 매우 클 때 발생하는 문제 
> 
> Activation Value의 값이 극단적인 값을 가지게 되고, Gradient의 값이 모두 0으로 수렴할 것이다. 
> 
> 이런 초기값 문제에 대해서 ‘Xavier Initialization’ 이라는 논문이 제시되었는데, 일단 Activation Function이 Linear하다는 가정 하에 아래과 같은 식을 사용하여 가중치 값을 초기화한다. 
> 
> 이 식을 이용하면 입, 출력의 분산을 맞춰줄 수 있게 된다. 
> 
> <img width="478" alt="스크린샷 2024-02-18 오후 2 05 08" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/4469daf1-af66-40e8-8095-a750c3b4d144">

> 
> 하지만 Activation Function을 ReLU로 정한 경우, 출력의 분산이 반토막나기 때문에 이 식이 성립하지 않는다. 
> 
> 보통 Activation Function이 ReLU인 경우, He Initialization을 사용한다. 
> 

### **Batch Normalization**

> 만약 Unit Gaussian Activation을 원하면, 현재 Batch에서 계산한 Mean과 Variance를 이용하여 정규화하는 과정을 모델에 추가해주는 것이다.
> 
> 
> <img width="464" alt="스크린샷 2024-02-18 오후 2 15 27" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/65759c66-baab-4a43-a5a9-ae650d4f3ded">
> 
> 각 Layer에서 가중치가 지속적으로 곱해 생기는 Bad Scaling의 효과를 상쇄시킬 수 있다.
> 
> <img width="464" alt="스크린샷 2024-02-18 오후 2 17 13" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/1939d094-083e-41b2-92df-f3d853de26ae">
> 
> Unit Gaussian으로 바꿔주는 것이 무조건 좋은 것은 아니다. 이에 유연성을 붙여주기 위해 **분산과 평균을 이용해 Normalizaed를 좀 더 유연하게 할 수 있다.** 
> 
> <img width="464" alt="스크린샷 2024-02-18 오후 2 17 36" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/e6048561-065d-4875-8d01-aca8afdf8c12">
> 
> Batch Normalization은 **Regularzation의 역할**을 할 수 있다. 즉, **Overfitting을 방지**할 수 있다. 
> 
> **가중치의 초기화 의존성에 대한 문제를 줄이고,** Test 할 때 Mini-Batch의 평균과 표준 편차를 구할 수 없으니 Training 하면서 구한 **평균의 이동 평균을 이용**해 고정된 Mean과 Std를 사용한다. 
> 
> 또한 **학습 속도를 개선**할 수 있다. 
> 

### **Hyperparameter Optimization**

> 딥러닝 모델을 만들 때, **고려해야 할 Hyperparameter들이 많다.**
> 
> 
> 보통 Train-Set으로 학습을 시키고 Validation-Set으로 평가를 한다. 
> 
> 만약 Hyperparameter를 바꿨는데, 업데이트 된 Cost 값이 원래 Cost의 값 보다 3배 이상 빠르게 증가할 경우 다른 파라미터를 사용해야 한다. 
> 
> <img width="460" alt="스크린샷 2024-02-18 오후 2 35 28" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/74422a84-cd91-4745-8d41-467b6ee8a5ab">
> 

> **Grid Search**
> 
> 
> 탐색의 대상이 되는 **특정 구간 내의 후보 Hyperparameter 값들을 일정한 간격을 두고 선정**하여, 각각에 대해 측정한 성능 결과를 기록한 뒤, **가장 높은 성능**을 발휘했던 **Hyperparameter 값을 선정**하는 방법이다. 
> 

> **Random Search**
> 
> 
> Grid Search와 큰 맥락은 유사하나, 탐색 대상 구간 내의 후보 Hyperparameter 값들을 **Random Sampling을 통해 선정한다는 점이 다르다.** 
> 
> Random Search는 Grid Search에 비해 불필요한 반복 수행 횟수를 대폭 줄이면서, 동시에 정해진 간격 사이에 위치한 값들에 대해서도 확률적으로 탐색이 가능하므로 최적 Hyperparameter 값을 더 빨리 찾을 수 있는 것으로 알려져 있다. 
> 

> **Hyperparameter Optimization 과정**
> 
> 1. Hyperparameter 값을 설정한다. 
> 2. 설정한 범위 내에서 파라미터 값을 무작위로 추출한다. 
> 3. 검증 데이터를 이용해 평가한다. 
> 4. 특정 횟수를 반복해 그 정확도를 보고 Hyperparameter 범위를 좁힌다. 
> 
> <img width="469" alt="스크린샷 2024-02-18 오후 2 40 20" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/34570353-7bae-4d50-b52d-90557ae6f1ce">
> 
> <img width="460" alt="스크린샷 2024-02-18 오후 2 34 56" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/8bf20074-0ec0-4733-bf4c-75d0869d19f3">
> 
> Hyperparameter를 정할 때 Loss Curve를 보고, Hyperparameter가 적합한지 아닌지 평가하는 경우가 많다. 
> 
> 만일 Loss Curve가 초기에 평평하다면 초기화가 잘못될 가능성이 크다. 
> 
> <img width="475" alt="스크린샷 2024-02-18 오후 2 38 06" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/17bb6494-03e4-460b-a415-5b0f006ef8ab">
> 
> Training Accuracy와 Validation Accuracy가 Gap이 클 경우, Overfitting 된 가능 성이 높다. 
> 
> Gap이 없을 경우 Model Capacity를 늘리는 것을 고려해야 한다. 
> 
> 즉, Training 한 데이터가 너무 작은 경우일 수도 있다. 
>