## **Lecture 7. Training Neural Networks 2**

### **Fancier Optimization**

> Vanilla Gradient Descent를 하게 되면 아래 식처럼 가중치를 초기화 해준다.
> 
> 
> 이때 **SGD는 Batch 단위로 끊은 것**이다. Mini-Batch 안에서 데이터 Loss를 계산하고, Gradient의 반대 방향을 이용해서 Parameter Vector를 업데이트 시켜준다. 
> 
> <img width="624" alt="스크린샷 2024-02-18 오후 4 37 55" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/3b31b6a0-b98b-4b41-b2e3-6276ac51f7b5">
> 
> **SGD 단점**
> 
> <img width="624" alt="스크린샷 2024-02-18 오후 4 41 57" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/a7a2c2e9-6d3f-48b2-ba1d-02fbdbac1540">
> 
> 손실 함수의 모양에 따라 영향을 많이 받는다. 
> 
> 위 그림처럼 타원 모양을 갖게 되면, 빨간 점에서 스마일 표시까지 어떻게 찾아가게 될까 ?
> 
> Loss가 수직 방향의 가중치 변화에 훨씬 더 민감하게 반응해서 빨간 선처럼 Gradient의 방향이 매우 크게 튀면서, 지그재그 형태로 지점을 찾아가게 된다. 
> 
> **가로 방향의 가중치는 느리게 변하는 반면 수직 방향의 가중치가 빠르게 변화하기 때문에 매우 느려진다는 단점**이 있다. 
> 
> 불균형한 방향이 존재한다면 SGD는 잘 동작하지 않는다. 
> 
> <img width="624" alt="스크린샷 2024-02-18 오후 4 45 35" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/fdac7a4e-5590-4d69-96af-0b9b72269e05">
> 
> <img width="624" alt="스크린샷 2024-02-18 오후 4 45 57" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/8092a1eb-cdb4-4bbb-87b7-f80d2203298f">
> 
> **Local Minima, Saddle Point** 문제도 있다. 
> 
> Local Minima는 기울기가 0이 되어버리기 때문에 찾았다 생각해 중간에 멈추는 것이다. 
> 
> Saddle Point도 순간적으로 기울기가 0에 가까운 지점이 있기 때문에 멈춰버릴 수도 있는 것이다. 보통 높은 차원에서는 Saddle Point가 더 일반적으로 발생한다. 
> 
> <img width="622" alt="스크린샷 2024-02-18 오후 5 00 44" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/1f4fc274-313a-4368-b14e-55954b6e6fb7">
> 
> Mini-Batch로 업데이트를 하는데, 이는 매번 정확한 Gradient를 얻을 수가 없다는 것을 의미한다. 대신 Gradient의 부정확한 추정값만을 구할 뿐이다. 
> 
> 따라서 **개선점이 필요**하다. 즉, **기울기가 0인 지점이 나와도 멈추지 않는 힘**이 필요하다. 
> 
> <img width="622" alt="스크린샷 2024-02-18 오후 5 03 41" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/13e2c4c1-176f-4dd8-87da-ee0b438ffefe">
> 
> 이 문제를 해결할 방법이 **SGD에 Momentum Term을 추가하는 것**이다. 
> 
> 쉽게 말해 **가속도를 주는 것**이다. 즉, **기울기가 0인 지점이 나와도 가속도로 인해 계속 갈 수 있게 하는 것**이다. 
> 
> 위 식에서 vx가 Velocity라고 생각하면 된다. 
> 
> rho는 마찰 계수로 너무 빠르게 가지 않도록 일종의 마찰을 주는 것이다. rho 값은 보통 0.9-.99를 주는 경우가 많다.
> 
> 이를 통해 Gradient Vector 방향 그대로 가는 것이 아니라, Velocity Vector의 방향으로 나아가게 된다. 
> 
> <img width="622" alt="스크린샷 2024-02-18 오후 5 07 20" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/0b426828-3d2b-432e-be9c-185f64c4d01c">
> 
> 위 파란 선과 같이 부드럽게 이동하는 것을 확인할 수 있다. 
> 
> 공이 굴러가는 것과 같은 속도로 생각하면, Gradient가 0이어도 충분히 움직일 수 있다. 
> 
> 그 덕분에 Local Minima와 Saddle Point를 극복할 수 있게 되고, 계속해서 내려갈 수 있게 된다. 
> 
> **Momentum을 추가함으로서 속도가 생기면 Noise가 평균화 되는 것**이다. 
> 
> <img width="622" alt="스크린샷 2024-02-18 오후 5 10 06" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/ca8e33fe-cba0-4378-83dd-9a1fe3860042">
> 
> 현재 Gradient 방향이 붉은색, 속도 Vector는 초록색을 의미한다. 
> 
> Actual Step은 이 둘의 가중 평균으로 구할 수 있다. 
> 
> **Nesterov Momentum**은 계산하는 순서를 조금 바꾼 것이다. 
> 
> 기존의 SGD Momentum은 현재 지점에서 Gradient를 계산한 뒤에 Velocity와 섞어준다. 
> 
> Nesterov Momentum은 **먼저 Velocity 방향으로 움직이고 그 지점에서 Gradient를 계산한다. 그리고 다시 원점으로 돌아가서 둘을 합치는 것**이다. 
> 
> Velocity의 초기값은 0으로 둔다. Nesterov Momentum은 Convex 최적화에서는 성능이 뛰어나지만, NN과 같은 Non-Convex에서는 그러지 못하다. 
> 
> Convex Function이 Optimal한 값을 찾기 쉽기 때문이다. 
> 
> <img width="625" alt="스크린샷 2024-02-18 오후 5 14 12" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/350898e9-f4e5-42e4-aa84-ab3d0b8884ec">
> 
> 위 사진과 같이 파란 부분이 기존 Momentum과 다르다. 파란 부분이 뜻하는 것은 **미리 Velocity 방향을 예측해서 Gradient를 구해준다는 의미**이다. 
> 
> <img width="625" alt="스크린샷 2024-02-18 오후 5 16 50" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/755f8465-3719-4694-bdb1-7aa4ffde147a">
> 
> 위 그림을 보면 Nesterov가 잘 동작한 것으로 나오는데, 실제로는 거의 그렇지 않다. 
> 
> <img width="625" alt="스크린샷 2024-02-18 오후 5 18 09" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/1468819e-8ba1-45f6-adbb-d9ac5bc17e31">
> 
> <img width="625" alt="스크린샷 2024-02-18 오후 5 18 23" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/66cd9a80-9b4d-49d7-8d97-609a52a3a16c">
> 
> <img width="625" alt="스크린샷 2024-02-18 오후 5 18 42" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/7bcf3098-6211-40ec-a002-0a301421e712">
> 
> **AdaGrad**는 **각각의 매개변수에 맞춤으로 갱신을 해주는 알고리즘**이다. 훈련 도중 계산되는 Gradient를 활용하는 방법이다. 
> 
> 여기서 A는 Adaptive로, 학습률을 조정하는 것이다. 
> 
> Velocity Term 대신 Grad Squared Term을 사용한다. 학습 도중에 계산되는 Gradient에 제곱을 해서 계속 더해준다. 
> 
> 업데이트 할 때에는 Update Term을 앞서 계산한 Gradient 제곱근 항으로 나눠준다. 
> 
> 나누다보면 분모 값이 커지기 때문에 Step을 진행할수록 값이 작아지게 된다. 
> 
> 그만큼 처음 올바른 지점으로 접근할 때 속도가 빨랐다 점차 속도가 느려진다는 것이다. 
> 
> 즉, 업데이트 속도가 느려지는 것이다. 
> 
> Convex 할 때에는 Minimum에서 서서히 속도를 줄여서 수렴하면 좋다. 
> 
> 하지만 Non Convex 할 때, Saddle Point에 걸려서 멈출 수도 없어 문제가 된다. 
> 
> 즉, 업데이트가 되지 않는 것이다. 
> 
> 이러한 문제의 해결책으로 RMSProp이 있다. 
> 
> <img width="625" alt="스크린샷 2024-02-18 오후 5 25 55" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/104c0bc5-0cff-4ab2-9f4f-8c0fad4219f0">
> 
> **RMSProp**은 **AdaGrad의 Gradient 제곱항을 그대로 사용**한다. 
> 
> 하지만 이 값들을 누적시키는 것 뿐 아니라 위의 파란 상자와 같이 **Decay Rate를 곱해준다.** 
> 
> Decay Rate 값은 보통 0.9-0.99를 사용한다. 
> 
> 현재 Gradient 제곱은 (1 - Decay Rate)를 곱해주고 더해준다. 
> 
> 이는 AdaGrad와 매우 비슷하기에 Step의 속도를 가속, 감속하는 것이 가능하다. 
> 
> 이를 통해 속도가 줄어드는 문제를 해결할 수 있다. 
> 
> <img width="625" alt="스크린샷 2024-02-18 오후 5 26 12" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/f2d1e8bf-c210-4f56-8280-a457be1153b4">
> 
> RMSProp이 원하는 이상적인 방향으로 가는 것을 확인할 수 있다. 
> 
> **Momentum과 AdaGrad 두 개를 합친 것이 있는데, 바로 Adam이다.** 
> 
> <img width="626" alt="스크린샷 2024-02-18 오후 5 32 10" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/2b684332-14c7-4848-ba2f-92dcd1bb936b">
> 
> **Adam**은 **First Moment와 Second Moment를 이용해서 이전의 정보를 유지**시킨다. 
> 
> 빨간 상자는 **Gradient의 가중합**이다. 파란 상자는 AdaGrad나 RMSProp처럼 **Gradient의 제곱을 이용하는 방법**이다. 
> 
> 하지만 초기 Step에서 문제가 발생한다.
> 
> 위 식을 보면 First Moment와 Second Moment가 0이다. 
> 
> 1회 업데이트 이후 Second Moment는 여전히 0에 가깝다. 이후 업데이트에서 Second Moment로 나누게 되는데 나눠주는 값이 작다보니까 분자가 커서 값이 튈 수도 있다. 
> 
> 값이 크면 Step이 커져서 이상한 곳으로 튈 수 있다. 이를 해결하기 위해 **보정항을 추가**한다. 
> 
> <img width="628" alt="스크린샷 2024-02-18 오후 5 37 20" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/278769ef-2e20-4aa7-aee3-6cb27b1f3086">
> 
> **1e - 7은 나누는 값이 0이 되는 것을 방지한다.** 
> 
> First, Second Moment를 업데이트하고, 현재 Step에 맞는 적절한 Bias를 넣어줘서 값이 튀지 않게 방지하는 것이다. 
> 
> <img width="628" alt="스크린샷 2024-02-18 오후 5 40 12" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/6f386db8-a97e-4a99-b254-169af9ee3338">
> 
> 위 그림과 같이 Adam이 효과가 좋고, 실제로 많이 사용한다. 
> 
> <img width="628" alt="스크린샷 2024-02-18 오후 5 40 54" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/c8372409-3884-4384-ada4-8cbcbf107f36">
> 
> 위에서 본 방법은 모두 Learning Rate를 가지고 있다. 
> 
> Learning Rate의 Hyperparameter 값을 찾는 것은 쉽지 않다. Learning Rate를 설정할 때, 다양한 방법이 있다. 
> 
> **Decay Learning Rate** 같은 경우 처음에 **Learning Rate를 높게 설정하고, 학습이 진행될수록 점점 낮추는 것**이다. 
> 
> <img width="628" alt="스크린샷 2024-02-18 오후 5 43 14" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/6e29e443-451c-45d8-b703-bba2b999a5f1">
> 
> 위 내용은 **ResNet** 논문에서 나온 내용으로, **Step Decay Learning Rate 전략을 이용해서 Loss를 나타낸 것**이다. 
> 
> 평평해지다 내려가는 구간은 Learning Rate를 낮추는 구간이다.
> 
> Learning Rate Decay는 Adam보다 Momentum을 사용한다. 
> 
> 보통 학습 초기에는 Learning Rate Decay가 없다고 생각하고, Learning Rate를 잘 선택하는 것이 중요하다.
> 
> Decay 없이 하다 필요한 구간이 어디인지 고려하는 방법도 좋다.
> 
> <img width="628" alt="스크린샷 2024-02-18 오후 5 45 46" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/ba892586-7f96-44e8-a8a7-925dee574ea4">
> 
> 지금까지 배운 것은 모두 1차 미분을 활용한 최적화이다.
> 
> <img width="628" alt="스크린샷 2024-02-18 오후 5 46 46" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/cbc4a3f1-cff0-4a03-b6a2-838166fc9c29">
> 
> 2차 근사의 정보를 추가적으로 활용하는 방법이 있다. 이를 이용하면 Mimima에 더 잘 근접할 수 있다. 이를 **Newton Step**이라고 한다. 
> 
> 2차 미분값들로 구성된 행렬인 **Hessian Matrix를 계산**한다. 
> 
> 이 Hessian Matrix의 역행렬을 이용하게 되면, 실제 손실 함수를 2차 근사를 이용해 Minima로 곧장 이동할 수 있을 것이다. 
> 
> <img width="628" alt="스크린샷 2024-02-18 오후 5 50 34" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/55e7894c-e01f-41db-bf44-ad853e71d1f0">
> 
> 기존과는 다르게 Learning Rate가 존재하지 않는다. 하지만 2차 근사도 완벽하지 않기 때문에 결국에는 필요하다. 그러나 NN에서는 사용할 수 없다. 
> 
> **Hessian Matrix는 NN Matrix이다. 따라서 실제로는 Hessian을 근사시켜 사용하는 방법을 사용한다. → L-BFGS**
> 
> <img width="628" alt="스크린샷 2024-02-18 오후 5 53 41" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/f4f91d22-1b1c-4a8d-a474-4aff72705fe5">
> 
> <img width="584" alt="스크린샷 2024-02-18 오후 5 54 25" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/14b778ef-ce80-4835-a6a5-c5849f2082d8">
> 
> Adam이 가장 좋은 선택이고, Full Batch라면 L-BFGS도 좋은 선택이 될 수 있다. 
> 
> 위 모든 것은 **Training Error를 줄이고, 손실 함수를 최소화 시키기 위한 방법**이다. 
> 
> <img width="626" alt="스크린샷 2024-02-18 오후 5 55 50" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/0cd7f8f1-72bb-4485-81fa-a68581dd22a3">
> 
> Loss를 줄이며, 동시에 Train과 Validation의 격차를 좁혀야 한다. **넓어지면 넓어질수록 Overfitting**이 되는 것이기 때문이다. 
> 
> Loss를 줄여 손실 함수를 최소화시키기 위해 Optimization 알고리즘을 사용한다. 
> 
> 최적화를 끝마친 상황에서 한 번도 보지 못한 데이터의 성능을 올리기 위해서는 어떻게 해야할까 ?
> 
> <img width="626" alt="스크린샷 2024-02-18 오후 5 58 01" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/85880b65-3043-462b-9ac5-34797a2af75b">
> 
> 가장 빠르고 쉬운 방법은 **Ensembles**이다. 
> 
> <img width="629" alt="스크린샷 2024-02-18 오후 5 59 32" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/97fecbe2-3a69-48b5-9e2a-d1a7d79a3131">
> 
> 다른 방법으로 모델을 독립적으로 학습시키는게 아니라, 학습 도중 중간 모델을 저장하고 Ensemble로 사용할 수 있다고 한다. 
> 
> 그리고 **Test 때** 여러 **Snap-Shot에서 나온 예측값들을 평균 내 사용**한다. 
> 
> Snap-Shot은 **구간을 정해 놓은 지점**이다. 훈련을 하는데, 10개의 Check Point를 두고, 10번째마다 새로 하겠다라는 식의 구간이라고 생각하면 된다. 
> 
> Ensemble을 사용하면 모델을 여러 개 만들기 때문에 그만큼 시간 소모가 된다. 
> 
> 그 방법 대신에 한 모델 안에서 10개의 구간을 두고 마치 Ensemble처럼 하겠다는 것이다. 
> 
> 위 사진의 빨간 부분을 보면 Train Loss가 낮아졌다 갑자기 올라가고 그러는데, 어느 지점에서 Learning Rate를 엄청 낮췄다 높였다를 반복하여 손실 함수가 다양한 지역에 수렴할 수 있도록 하는 것이다. 
> 
> Ensemble 기법으로 모델을 한번만 학습시켜도 좋은 성능을 얻을 수 있다. 
> 

### **Regularization**

> Ensemble은 모델을 여러 개 만들기 때문에 효율적이지 않다고 하는데, Ensemble을 사용하지 않고 모델 성능을 향상시킬 수 있는 방법은 어떤 것이 있을까 ?
> 
> 
> <img width="629" alt="스크린샷 2024-02-18 오후 8 01 54" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/cf6da7c3-c47e-4362-ab30-559504685b6c">
> 
> **단일 모델의 성능을 올리는 것이 목적**이고 L1과 L2는 뉴런 네트워크에서 실제로는 잘 사용하지 않는다. 
> 
> <img width="629" alt="스크린샷 2024-02-18 오후 8 03 25 (1)" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/93812652-e852-4a87-b4c4-802d087979c1">
> 
> 때문에 사용하는 Regularization이 **Dropout**이다. 
> 
> Forward Pass 과정에서 **일부 뉴런을 0으로 만드는 것**으로 오로지 뉴런의 일부만을 사용한다. 
> 
> Forward Pass 반복마다 그 모양은 계속 바뀐다. 현재의 Activation 일부를 0으로 만들어 다음 Layer의 일부가 0과 곱해지게 하는 것이다. 
> 
> <img width="629" alt="스크린샷 2024-02-18 오후 8 06 53" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/4b0c1e8a-8e93-4486-acd8-fb8ba0942bdc">
> 
> Dropout의 특징은 **Feature 간의 상호작용을 방지하는 것**이다. 
> 
> 이후 모델이 고양이라고 예측할 때 다양한 Feature를 골고루 이용할 수 있도록 한다. 
> 
> 따라서 Dropout이 **Overfitting을 어느정도 방지**해준다. 
> 
> 단일 모델로 Ensemble 효과를 가질 수 있다. **Dropout은 아주 거대한 Ensemble 모델을 동시에 학습시키는 것과 같다.** 
> 
> 즉, **Forward Pass마다 Dropout을 Random하게 하니까 Forward Pass마다 마치 다른 모델을 만드는 것처럼 효과가 나오게 될 수 있다.** 
> 
> <img width="629" alt="스크린샷 2024-02-18 오후 8 11 57" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/92976913-7758-4f2c-8507-8fa8d43e7aba">
> 
> z는 Random이다. Test Time에 임의의 값을 부여하는 것은 좋지 않다. 
> 
> <img width="629" alt="스크린샷 2024-02-18 오후 8 13 01" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/2f7df06a-187c-4a76-8139-76f4e446deec">
> 
> <img width="632" alt="스크린샷 2024-02-18 오후 8 13 27" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/7c53a391-76b1-4fd5-88ee-aae99d5b5e44">
> 
> Dropout = 0.5로 학습시킨다고 생각할 때, 4가지 경우의 수가 존재하고 그 값들을 4개의 Mask에 대한 평균화 시켜준다. 
> 
> 이 부분에서 Train, Test 간 기대값이 서로 상이하다. 이를 해결하기 위해 Dropout Probability를 네트워크의 출력에 곱한다. → 기대값이 같아진다. 
> 
> 일부 Node를 무작위로 0으로 만들어주고, Test Time에는 하나의 값을 곱해주면 된다. 
> 
> <img width="632" alt="스크린샷 2024-02-18 오후 8 16 02" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/1df7a92d-f610-4546-bd18-82372e523a9a">
> 
> Test Time에는 기존의 연산을 가져가고, Train Time에서 p를 나눠준다. 
> 
> **Dropout을 사용하게 되면 전체 학습 시간은 늘어나지만 모델이 수렴한 후 더 좋은 일반화 능력을 얻을 수 있다.** 
> 
> 즉, Train Time에는 네트워크에 무작위성을 추가해 Train-Set에 너무 Fit 하지 않도록 한다. 
> 
> Test Time에는 Randomness를 평균화 시켜서 Generalization 효과를 주는 것이다. 
> 
> BN도 비슷한 역할을 할 수 있으므로 **BN 시에는 Dropout을 하지 않는다.** 
> 
> Regularization이 충분하기 때문이다. 
> 
> <img width="632" alt="스크린샷 2024-02-18 오후 8 18 52" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/bb11835f-6a16-4f50-beaa-35f6c15eacde">
> 
> Train Time에 이미지를 무작위로 변환시켜 볼 수 있다. Lable을 그대로 놔둔 채 진행하고 이미지를 반전시킨다. 
> 
> <img width="632" alt="스크린샷 2024-02-18 오후 8 20 08" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/bec459e1-e260-472d-b010-f91e6101fc0d">
> 
> 이미지를 잘라내고 반전시켜 학습한다. 또는 이미지의 밝기를 낮추기도 한다. → Color Jittering
> 
> Train Time 입력 데이터에 임의의 변환을 시켜주게 되면 일종의 Regularization 효과를 얻을 수 있다. 
> 
> Train Time에는 Stochasticity가 추가되고 Test Time에는 Marginalize Out 되기 때문이다. 
> 
> <img width="632" alt="스크린샷 2024-02-18 오후 8 22 55" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/a5477fc7-fa8f-4cef-a061-9aa4e86ac9ac">
> 
> Dropout과 유사하게 **Dropconnect**라는 방법이 있는데, 이는 Activation이 아닌 **Weight Matrix를 임의적으로 0으로 만들어주는 방법**이다. 
> 
> <img width="632" alt="스크린샷 2024-02-18 오후 8 24 37" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/9d7011dd-7569-4297-b170-05d4f635559b">
> 
> Fractional Max Pooling은 Pooling이 될 지역을 임의로 설정해 Sampling 하는 것이다. 
> 
> <img width="632" alt="스크린샷 2024-02-18 오후 8 25 29" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/29efda12-3e3a-45ce-8d0c-08447c3a18b9">
> 
> Stochastic Depth
> 
> Train Time에서 일부 Layer를 제외 학습하고, Test Time에는 전체 네트워크를 다 사용한다. 
> 
> 보통 BN만으로 충분하지만, Overfitting이 발생하는 경우 Dropout을 추가하기도 한다. 
> 

### **Transfer Learning**

> **Transfer Learning**은 **원하는 양보다 더 적은 데이터를 가진 경우 사용하는 방법**이다.
> 
> 
> Overfitting이 일어날 수 있는 상황 중 하나가 충분한 데이터가 없을 때이다. 
> 
> <img width="632" alt="스크린샷 2024-02-18 오후 8 28 34" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/3c7891a4-d17d-429e-a088-d309b990058b">
> 
> **가장 마지막 FC Layer는 최종 Feature와 Class Score 간의 연결**인데, 이를 초기화시킨다. 
> 
> 그리고 차원을 줄이고, 마지막 Layer만 가지고 데이터를 학습시킨다. 데이터가 조금 많다고 생각되면 전체 Fine Tuning을 해볼 수도 있다. 
> 
> <img width="632" alt="스크린샷 2024-02-18 오후 8 30 14" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/cf268f0a-672b-4935-b774-a671775c9ff4">
> 
> <img width="632" alt="스크린샷 2024-02-18 오후 8 30 37" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/39733820-07b4-4ea0-8c2c-8e02311ccf03">
>