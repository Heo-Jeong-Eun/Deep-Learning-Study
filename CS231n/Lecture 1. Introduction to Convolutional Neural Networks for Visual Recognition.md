## **Lecture 1. Introduction to Convolutional Neural Networks for Visual Recognition**

### **A Brief History of Computer Vision**

> 과거 생명체의 종이 짧은 시간 내 다양화 된 **Cambrian Explosion, Evolution’s Big Bang**의 원인을 **Vision의 발달**로 생각한 신경 과학자들은 생명체가 **시각적으로 물체를 인식하는 프로세스**에 대해 연구하기 시작했다.
> 
> 
> 그 결과 일차 시각 피질 부분에는 다양한 세포가 존재하지만, 가장 중요한 세포는 어떤 방향으로 이동할 대 방향의 가장자리, 즉 **Oriented Edges**에 반응하는 단순 세포라는 것을 알게 되었다. 
> 
> **이미지 처리는 방향의 가장자리라는 단순한 구조에서 시작해 정보가 시각적 처리 경로를 따라 이동하며 뇌가 복잡한 구조를 인식할 수 있을 때까지 시각적 정보의 복잡성을 증가시키는 방식으로 이루어져 있다.** 
> 
> David Marr는 MIT의 CV 과학자로, CV 알고리즘 개발에 대한 책을 썼다.
> 그는 **이미지로 완전한 3D 표현까지 도달을 위해서는 몇 단계의 과정**이 필요하다고 주장했다.
> 
> <img width="692" alt="스크린샷 2023-12-14 오후 7 31 41" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/f3107941-79a4-4089-bb7f-3d9c693cc43a">
> 
> 첫 번째 단계는 **Primal Sketch**로 가장자리, 모서리(Edge), 막대(Bar), 끝(Ends), 가상의 선(Virtual Line), 곡선(Curves), 경계(Boundaries)가 이에 해당되고, 신경 과학자들에게 영감을 받았다. 
> 
> 두 번째 단계는 **Two-and-a-Half-D Sketch**로 표면(Surface) 정보, 깊이(Depth) 정보, Layer, 시각 장면을 구성하는 불연속 점과 같은 것들을 종합한다. 
> 궁극적으로 모든 것을 한 곳에 모아 표면부터 Volumetric Primives(체적 초기 단계) 등 계층적으로 구성된 최종적인 3D 모델을 만들어 낸다. 
> 
> 이후 사람들은 어떻게 Block World를 뛰어 넘어 **실제 세계를 인식하고 표현**할 수 있을지에 대한 생각을 했고, 두 가지 아이디어가 제안되었다. 
> 
> <img width="684" alt="스크린샷 2023-12-14 오후 7 33 05" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/250b38c9-b01b-4c43-a8e6-61c713141b40">
> 
> 두 개념은 **모든 객체를 단순한 기하학적 원형으로 구성** 하는 것에서 공통점을 가진다. 
> 
> 계속해서 CV에 대한 연구가 이어졌지만, 객체 인식 문제 해결에 어려움이 있었고, 이를 해결하기 위해 **이미지를 가져와 픽셀을 의미 있는 영역으로 그룹화하는 작업인 Segmentation**이 탄생하게 되었다. 
> 
> 이후 David Lowe의 특징 기반 물체 인식, **SIFT Features**이 등장했다. David Lowe는 물체의 특징 중 일부는 다양한 변화에 강인하고 불변하다는 점을 이용해 물체에서 중요한 특징을 찾아내고, 비슷한 물체와 특징을 매칭시켰다. 
> 
> <img width="671" alt="스크린샷 2023-12-14 오후 7 41 41" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/717203b0-dd46-4e7f-abf0-e55e5b2d6326">
> 
> 이미지 특징을 사용하면서 **전체 장면 인식도 가능**해졌는데, 대표적인 예로 Sparial Pyramid Matching이라는 알고리즘이 있다. 
> **이미지 내의 여러 부분과 해상도에서 추출한 특징을 하나의 특징 기술자로 표현, Support Vector 알고리즘을 적용**한다. 
> 
> <img width="591" alt="스크린샷 2023-12-14 오후 7 43 54" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/a4b76203-8251-44fe-a99a-077993238161">
> 
> **사람 인식**과 관련된 연구는 주로 어떻게 해야 사람의 몸을 현실적으로 인식하고, 이미지로 모델링 할 수 있을까에 대한 연구였다. 대표적으로 **Histogram of Gradients**와 **Deformable Part Models**라는 연구가 있다. 
> 
> <img width="669" alt="스크린샷 2023-12-14 오후 7 48 42" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/79bf6e5f-168e-4c99-b24d-a93828e46295">
> 
> 2000년대 이후 PASCAL, ImageNet의 Challenge 등 여러 프로젝트가 등장했고, 이미지 인식은 점점 더 발전했다. 
> 
> <img width="677" alt="스크린샷 2023-12-14 오후 7 50 47" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/4d9cd2c5-400b-49a0-986d-2ec88914e1ba">
> 
> 특히 2012년 **오차율이 10% 가까이 하락**했는데, 이때 우승한 알고리즘이 **CNN**이다. 
> 
> 이후 CNN은 사람의 오차율보다 낮은 수치를 기록하게 된다. 
> 
> <img width="653" alt="스크린샷 2023-12-14 오후 7 51 18" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/ba707077-75f5-4614-9990-8132a149b3a3">
> 

### **CS231n Overview**

> CS231n 강의는 **Image Classification**에 초점을 두고 있다. 이와 관련된 **Image Recognition** 문제는 무궁무진하다.
> 
> 
> <img width="681" alt="스크린샷 2023-12-14 오후 8 02 06" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/0c88bcbc-8b87-4d8a-adc2-c7b95bbd2061">
> 
> 2010년에서 알고리즘에서 주목해야 할 점은 **여전히 계층적이고, 가장 자리를 감지하며 불변의 개념을 갖고 있다는 것**이다. 
> 
> 2012년 **7개의 Layer를 가진 CNN이 개발**되었고, 현재 **AlexNet**으로 알려져있다. 이후 **ImageNet의 우승 알고리즘은 신경망**이었다.  
> 
> <img width="681" alt="스크린샷 2023-12-14 오후 8 00 31" src="https://github.com/Heo-Jeong-Eun/Deep-Learning-Study/assets/60500256/4f1f4f22-ced1-43ee-8737-5ea66c298f14">