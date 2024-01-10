## **Lecture 3. Loss Functions and Optimization**

### **Loss Function**

> 최적의 W값을 구하기 위해 **카테고리 스코어 값의 차이를 정량화하는, 즉 W가 좋은지 안좋은지 판단할 방법**이 필요하다.
> 
> 
> 이미지 분류 시, 어떠한 수식에 의해 출력된 스코어에 대해 불만족하는 정도를 정량화 시킬 수 있는 것이 **Loss Function, 손실 함수**이다. 
> 
> 손실 함수 L_i를 정의하면 예측함수 f의 Train-Set을 얼마나 안좋게 예측하는지를 정량화 시킨 값을 준다. 
> 
> 최종 손실 값인 L은 전체 데이터에서 각 N개의 샘플에 대한 Loss 평균이 된다. 
> 
> <img width="709" alt="스크린샷 2024-01-07 오후 8 00 56" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/95b5e827-5b13-44ea-9858-ca91b500eeba">
> 
> f는 입력 이미지 x와 가중치 행렬 W를 입력으로 받아 새로운 테스트 이미지에 대해 y를 예측한다. 
> 
> 딥러닝 알고리즘은 어떤 x와 y가 존재하고, 가중치 W가 얼마나 좋을지를 정량화하는 손실 함수를 만드는 것이다. 
> 

> **Multi-Class SVM Loss, Support Vector Machine**
> 
> 
> SVM은 2, 3차원 이상의 고차원인 초평면에서 결정 경계를 기준으로 데이터를 분리한다. 
> 
> 이때 **결정 경계와 인접한 데이터 포인트를 Support Vector, 이 Vector와 결정 경계 사이의 거리를 Margin**이라고 한다. 
> 
> SVM은 허용 가능한 오류 범위 내에서 가능한 최대 Margin을 만들어야 한다. 
> 
> <img width="704" alt="스크린샷 2024-01-07 오후 8 06 25" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/574d87f5-06ae-40ed-b34a-19a1b0d1d1b9">
> 
> <img width="711" alt="스크린샷 2024-01-07 오후 8 06 42" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/6877a444-3694-4a24-8087-c1c5d6b93aea">
> 
> <img width="711" alt="스크린샷 2024-01-07 오후 8 07 00" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/e6e62c4a-f76b-4e4f-a4e5-4fbeb9a337bc">
> 
> - SVM은 데이터에 둔감하고, 카테고리 스코어 값이 중요한게 아니라 **정답 클래스가 다른 클래스보다 큰지, 작은지가 중요**하다.
> - Loss의 최소값은 0이고, 무한대까지 최대 값을 가진다. 또한 **Loss는 클래스 갯수 - 1만큼 나온다.**
> - W는 Train-Set에 맞춰져 있기 때문에 Test-Set에서는 Train-Set에서 구한 W와 다를 수 있다.
>     
> <img width="668" alt="스크린샷 2024-01-07 오후 8 12 05" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/ced37de3-ee42-4d5f-8f3e-5389ca52cb9c">
>     

> 학습된 W는 Train-Set에 맞춰져 있다. 과도한 Train-Set 학습으로 인해 Test-Set를 맞추지 못해 정확도가 떨어지는 것을 **Overfiting, 과적합**이라고 한다. 따라서 **Test-Set에도 알맞은 W를 찾아주는 것이 필요**하다.
> 
> 
> 과적합을 방지하기 위해 **Regularization, 정규화 항을 추가해 사용**한다. 
> 
> Data Loss Term에서는 모델이 Train-Set에 잘 맞으며, Regularization Term을 추가해 모델이 좀 더 단순한 W를 선택하도록 도와준다. 
> 
> <img width="616" alt="스크린샷 2024-01-07 오후 8 13 26" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/05c66852-0f1b-45a3-8034-737b43442523">
> 
> **Regularization은 Train-Set에만 맞는 가중치 W를 학습하려고 할 때, 어느정도 Penalty를 부여한다.** 
> 
> 가중체에 대한 선호도 표현과, 모델을 단순하게 만들어 Test-Set에서도 작동이 가능하며 곡면성을 추가하여 최적화 향상을 위해 규제가 필요하다. 
> 
> <img width="703" alt="스크린샷 2024-01-07 오후 8 23 57" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/c36d3ecc-46e9-492f-a65f-64e30b62bd8e">
> 
> 가장 보편적인 Regularization으로는 **L2 Regularization**이 있다. 
> 
> L2 Regularization은 Weight Decay라고도 한다. **복잡도를 상대적으로 W1, W2 중 어느 값이 더 매끄러운지를 측정**한다. x의 특정 요소에 의존하기 보다 x의 모든 요소가 골고루 영향을 미치길 원할 때 L2를 사용한다. 
> 
> <img width="688" alt="스크린샷 2024-01-07 오후 8 31 06" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/c1707d14-941f-483b-a237-007f1caa97eb">
> 
> - L1 : W에 0이 아닌 요소가 많으면 복잡하고, W에 0이 많으면 덜 복잡하다.
> - L2 : W가 어느 쪽에 치중되어 있으면 복잡하고, W에 요소가 전체적으로 분산되어 있을 때 덜 복잡하다.

> **Softmax Classifier, Multinomial Logistic Regression**
> 
> <img width="690" alt="스크린샷 2024-01-09 오후 12 59 30" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/b2d2b26f-344a-4cda-a303-06b3fc38a9ef"> 
> 
> SVM Loss에서는 스코어 자체를 신경쓰지 않고 Safety Margin을 포함해 높은지 낮은지만 판단했는데, **Softmax Classifier의 손실 함수는 스코어 자체에 추가적인 의미, 즉 확률 분포를 부여**한다. 
> 
> **Softmax**라는 함수를 사용하는데, **스코어를 가지고 클래스 별 확률 분포를 계산**한다. 
> 
> Softmax 함수는 스코어를 모두 가지고 스코어들에 지수를 취해 양수로 만들고, 그 지수들의 합으로 다시 정규화를 진행한다. 
> 
> 따라서 **Softmax 함수를 거치게 되면 확률 분포를 얻을 수 있고, 이는 해당 클래스일 확률을 나타낸다.** 확률이므로 0-1 사이의 값이며, 모든 확률의 합은 1이 된다. 
> 
> 연산 과정은 주어진 스코어에 대해 exp 연산을 취해 그 값을 구하고, 그 값을 모두 더한 값을 해당 클래스 스코어에 나눠준다. 마지막으로 -log 연산으로 최종 확률을 계산한다. 따라서 Loss는 **-log(정답 클래스일 확률)**이다. 
> 
> <img width="713" alt="스크린샷 2024-01-09 오후 1 17 34" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/95ef20c7-160c-4cd3-9c20-a406fb8743a5">
> 
> SVM VS Softmax
> 

### **Optimization**

> 손실 함수가 주어진 W값에 대해 좋은지 안좋은지 정량화 할 수 있도록 도와준다. 이때 **어떤 효율적인 과정을 통해 W가 될 수 있는 모든 경우의 수를 찾아보고 가장 최선의 W가 무엇인지 알아내는 과정**이 **최적화 과정**이다.
> 
> 
> 즉, **Loss를 최소화 시킬 수 있는 W를 찾는 것**으로 **Random Search**와 **Gradient Descent**가 대표적이다. 
> 
> **Random Search**는 W의 위치 값 포인트를 무작위로 찾는 방법이고, **예측의 정확도가 15-95%로 불안정**하다. 따라서 사용하면 안되는 방법이다. 
> 
> 실질적으로는 **Gradient Descent, 경사 하강법**을 사용한다. 
> 
> <img width="718" alt="스크린샷 2024-01-10 오후 3 58 27" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/0b5d6cde-e6cf-4b45-96a5-e1ba3019ec43">
> 
> **경사, Slope**은 **1차원 공간에서 어떤 함수에 대한 미분 값**이다. 따라서 경사는 **방향 정보**를 가지고 있다. 
> 
> x를 입력으로 받으면, 출력은 곡선의 높이로 생각할 수 있다. 곡선의 일부를 구하면 기울기를 계산할 수 있다. 
> 
> **최소 반지름이 0으로 수렴하는 아주 좁은 근방에서는 가장 가파르게 감소하는 최선의 방향을 구할 수 있고, 이 방향을 따라 가중치 Vector를 움직인다.** 
> 
> **x가 Vector일 경우, 이 미분을 편미분이라고 한다.** 
> 
> **Gradient**는 **각 차원으로 편미분들을 모아 놓은 Vector의 집합**이다. 
> 
> **Gradient의 각 요소**는 임의의 방향으로 갈 때 **함수 f의 경사가 어떤지**에 대한 정보를 알려준다.
> 
> <img width="727" alt="스크린샷 2024-01-10 오후 4 05 35" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/324c1ce0-65c4-419d-bbbf-846b0f21e1c6">
> 
> Gradient의 각 요소는 한 방향으로 아주 조금씩 이동했을 때, Loss 값이 어떻게 변하는지 알려준다. 
> 
> 하지만 이 방법은 매우 오랜 시간을 필요로 한다. 
> 
> 따라서 Numerical Gradient를 사용하지 않고, **Analytic Gradient를 사용**한다. 
> 
> - Numerical Gradient → Approximate(근사치), Slow, Easy to Write
> - Analytic Gradient → Fast, Exact, Error-Prone
> 
> **Gradient를 나타내는 식을 찾고, 이를 이용해 한번에 Gradient dW를 계산한다.** 
> 
> 하지만 구현 시 실수하기 쉽기 때문에, 실제 응용할 때 해석적으로 구한 뒤 수치적으로 구한 것과 비교 후 틀린 부분을 고치는 작업을 하고 이를 **Gradient Check**라고 한다. 
> 
> <img width="626" alt="스크린샷 2024-01-10 오후 4 12 41" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/59603af5-e6ac-4a61-86db-fb4867a808e7">
>
> <img width="679" alt="스크린샷 2024-01-10 오후 4 10 00" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/9a02a12f-ac39-4fce-a790-a764c56bef35">
> 
> Data-Set의 Loss를 계산하는 것은 많은 시간을 필요로 한다. 수백만번의 계산이 필요하기 때문이다. 
> 
> 손실 함수는 각각 학습 데이터를 분류기가 얼마나 안좋게 분류하는지를 계산해 전체 Train-Set Loss 평균을 전체 Loss로 사용했다. 
> 
> 실제로는 **Stochastic Gradient Descent**를 사용한다. 
> 
> **전체 Data-Set의 Gradient와 Loss를 계산하지 않고, Mini-Batch라는 작은 학습 데이터 집합으로 나누어 학습하는 것**이다. 
> 
> **Mini-Batch**는 **보통 2의 승수로 32, 64, 128을 자주 사용**한다. 따라서 이 작은 Mini-Batch를 이용해 **Loss 전체 합의 추정치**와, **실제 Gradient의 추정치**를 계산한다. 
> 
> <img width="703" alt="스크린샷 2024-01-10 오후 4 22 12" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/8f249ab6-489b-4a91-b852-2c985cf9a4d6">
> 
> <img width="643" alt="스크린샷 2024-01-10 오후 4 22 43" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/4dd2dc73-d49b-4629-8377-e0a087965dac">
> 
> http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/
> 

### **Image Features**

> Linear Classifier는 실제 Raw 이미지 픽셀을 입력으로 받았는데, 이는 Multi-Modality의 경우를 생각하면 좋은 방법이 아니다. 애초에 영상 자체를 입력받는 것은 성능상 좋지 않다.
> 
> 
> 이미지에 Linear Classifier을 적용하는 것이 아닌, **단순 Histogram이나 코드 형태로 Feature를 추출**하고, **추출한 Feature를 Concat**으로 이어준다. 이후 Linear Classifier를 적용하는 것이 일반적인 방법이다. 
> 
> <img width="684" alt="스크린샷 2024-01-10 오후 5 05 58" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/b9e37151-9744-4b43-bc8f-4756fd8cb1f7">
> 
> 아래 그림의 왼쪽의 경우 Linear한 경계를 그릴 수 있는 방법이 없다. 하지만 오른쪽과 같이 극좌표계를 바꾸는 방법 등으로 적절하게 특징 변환을 한다면, 복잡하던 데이터가 변환 후 선형 분리가 가능하게 바뀐다. 
> 
> <img width="640" alt="스크린샷 2024-01-10 오후 5 07 05" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/50a085b9-f9c2-41ad-87c9-78050efcd02d">
> 
> Image Feature → Motivation 
> 
> 특징 변환의 다른 예시로 **Color Histogram**이 있다. 이미지 내의 **컬러를 모두 픽셀로 파악**하고, 그 컬러의 픽셀을 **전체 파노라마에서 Color Bin이 몇개인지 갯수를 세어 Feature를 추출하는 방법**이다. 
> 
> <img width="640" alt="스크린샷 2024-01-10 오후 5 11 31" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/40a2b806-3c2b-41f4-a46a-9e230082d211">
> 
> 또 다른 특징 Vector에는 **Histogram of Oriented Gradients, HoG**가 있다. 
> 
> **8*8 픽셀로 구성된 구역을 총 9가지 Edge의 Bin으로 나눠 9가지의 Bin에 몇 개가 속한느지 Edge의 의존도를 추출**하는 것이다. 
> 
> <img width="589" alt="스크린샷 2024-01-10 오후 5 14 24" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/e8399cf1-4063-4069-a123-fb729091562d">
> 
> 마지막으로 언어 처리에서 자주 사용되는 기법인 **Bag of Words**가 있다. 
> 
> 이미지의 여러 지점을 보고, 이 작은 지점의 **Patch를 Frequency나 Color의 Vector로 기술**한다. 이것을 **사전화**해 사전 내에서 테스트할 이미지와 **가장 유사한 Feature Vector**를 찾는다. 
> 
> <img width="722" alt="스크린샷 2024-01-10 오후 5 18 31" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/60d42c58-f834-4442-95bc-e7cfe23f7bcf">
> 
> 위에서 설명한 3가지 **기존 방식은 임의적으로 Feature Extraction을 통해 이미지를 분류하는 방법**이다. 
> 
> > <img width="672" alt="스크린샷 2024-01-10 오후 5 18 14" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/2700776e-86d9-4090-aad1-521a918e0ead">