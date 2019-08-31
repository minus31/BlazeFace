# BlazeFace : Sub-millisecond Neural Face Detection on Mobile GPUs

> *논문 링크 : https://arxiv.org/pdf/1907.05047v2.pdf*

이 논문에서는 모바일 GPU를 사용하는 서비스, 추론 환경에서 가볍고 좋은 성능을 내는 작동하는 얼굴 검출 프레임워크, **"BlazeFace"** 를 제안했습니다. 

결과부터 확인하면, MobileNetV2기반의 SSD 보다, Average-Precision 기준 약 0.66% 높고, Inference Time은 1.5ms 만큼 더 낮습니다. 

![](https://www.dropbox.com/s/9we872zjvyma79m/Screenshot%202019-08-14%2000.59.57.png?raw=1)

이러한 빠른 추론 속도와 좋은 성능 덕분에, 저자들은 증강현실, 2D/3D 얼굴 키포인트 검출, Facial expression 분류, Surface geometry 추정 문제 등 다양한 객체 검출 관련 문제에 응용 될 수 있다고 말합니다. 다만, 본 논문은 모바일 환경에서의 얼굴 검출 문제에 집중하였습니다. 

#### BlazeFace의 특징

BlazeFace의 특징은 크게 3가지로 정리 할 수 있습니다. 

1. MobileNet 기반의 구조 
2. GPU에 친화적으로 구성한 Anchor(네트워크로 조정하게 될 미리 정의된 바운딩 박스)

3. 후처리로 tie resolution strategy를 사용

하나씩 자세히 설명하겠습니다. 

#### 네트워크 구조

*BlazeFace*의 구조는 MobileNet V2의 구조를 차용했습니다. Depth-wise separable convolution 연산과 Residual connection은 그대로 사용되었습니다. 여기서 중요한 차이점은 2 가지 입니다. 

1. **수용영역(Receptive filed)의 확대**

   MobileNet을 포함한 최근에 나오는 Convolutional 네트워크에서는 대부분 **3x3** 크기의 가중치 행렬을 사용하지만, *BlazeFace* 에서는 Convolution 연산에 **5x5** 크기의 가중치 행렬을 사용합니다. 

   저자들은 연산량의 이득을 보기 위해서 사용하는 **Depth-wise separable convolution** 연산에서 가중치 갯수에 더 영향을 미치는 것이 Point-wise convolution 연산이라는 사실에 주목했습니다. 

   가령, $\large s \times s \times c$ 의 입력이 들어오고, 크기가 $\large k \times k$ 인 가중치 행렬을 사용했을 때, Depth-wise convolution 연산에 필요한 가중치 갯수는 $\large s^2 c k^2$ 이고. Point-wise convolution 연산의 경우 출력하는 채널 수를 $\large d$ 라고 할  때 $\large s^2cd$ 개의 가중치가 필요합니다. (Batch-normalization을 사용하기에  Bias는 고려하지 않았습니다.) 이를 보면 Point-wise convolution 연산은 Depth-wise convolution의 가중치 갯수의 $\dfrac{d}{ k^2}$ 배 만큼의 가중치를 가진다고 할 수 있습니다.  보통의 경우 $k^2$이 출력 채널의 갯수보다 작으므로 Point-wise convolution에 사용되는 가중치가 더 많을 것 입니다. 이는 Depth-wise convolution에서 더 큰 가중치 행렬을 사용하는 것이 더 많은 채널을 사용하는 것보다 비교적 더 저렴하다는 것을 뜻 합니다. 또한, 큰 가중치 행렬을 사용하는 것은 특정 크기의 수용 영역(Receptive field) 크기에 도달하기 위해 필요한 레이어의 갯수를 줄일 수 있기 때문에 *BlazeFace*에서는 5x5 크기의 가중치 행렬을 기본으로 사용합니다. 

2. **Bottleneck 구조** 

   MobileNet V2에서의 Bottleneck layer는 입력의 채널 수를 늘리고 $\rightarrow$ 출력의 채널 수를 감소 하는 구조 였지만, *BlazeFace*에서는 Bottleneck layer의 중간의 채널 수 경량화를 위해 입력 채널 수를 감소하고 $\rightarrow$ 출력 채널을 늘려서 사용합니다.(*BlazeFace*에서는 주로 입력을 24개의 채널로 감소하고 48 또는 96 개의 채널을 출력했습니다.)

언급한 두 가지의 차이점이 반영된 것을 **BlazeBlock** 이라 표현합니다. 다음 그림처럼 Convolution 횟수와 Bottleneck 유무를 기준으로 Single Blazeblock 과 Double Blazeblock으로 나뉩니다. 

![](https://www.dropbox.com/s/39061on4xqumdmh/Screenshot%202019-08-14%2001.44.09.png?raw=1)

전체 네트워크 구조를 조망해보면, 128x128 크기의  RGB 이미지를 입력 받아 5개의 Single Blazeblock과 6개의 double Blazeblock을 거쳐 특징을 추출 합니다. 다음 표에서 전체 구조를 확인 할 수 있습니다. 

![](https://www.dropbox.com/s/f3p8bpa8bspqmky/BlazeFaceNetwork.png?raw=1)

##### Anchor 구성 

*BlazeFace*에서도 SSD와 마찬가지로 미리 정의된 바운딩 박스를 사용하여 객체의 위치를 조정합니다. 이 역시도 차이점이 있습니다. 

 GPU 사용에 있어, 비용적인 측면에서 CPU와 두드러지게 차이나는 연산 중 하나가 특정 레이어에 대한 연산을 배분하는 연산이라고 합니다. 그리고 *[Pooling Pyramid Network](https://arxiv.org/abs/1807.03284)(Pengchong Jin et al. 2018)* 논문에서 제시한 방법은( 여러 feature map 크기에서 따로 Bounding box 예측을 하는 것이 아니라 공유된 convolution 가중치를 사용하여 여러 크기의 피쳐맵의 예측을 수행하는 것) 특정한 해상도 이하의 피쳐맵이 꼭 필요하지 않음을 시사했습니다. 

 위와 같은 이유로 BlazeFace에서는 **8x8 크기 이하로 그리드 크기를 줄이지 않습니다**. 그리고 2x2, 4x4, 8x8 크기의 피쳐맵에서 각각 2개의 anchors를 8x8 크기 피쳐맵에서의 6개의 anchors로 대체하였습니다. 

 아래 그림처럼, 결과적으로 16x16 크기의 피쳐맵에서 각 픽셀마다 2개의 anchors, 8x8 크기의 피쳐맵에서 6개의 anchors를 사용하여 예측을 합니다. 

![](https://www.dropbox.com/s/yaklvn29vowk1og/Screenshot%202019-08-14%2002.56.09.png?raw=1)

얼굴 검출이라는 문제 특성 상, 가로세로비는 1:1 고정 하였습니다. Scale에 관련해서는 논문에 정확히 언급하지 않을 것으로 보아 SSD의 방법을 그대로 유지하는 것으로 추정됩니다. BlazeFace에서는 총 896개의 바운딩 박스를 사용해 객체를 검출 합니다. 

##### 후 처리

 BlazeFace에서는 8x8 이하 크기의 피쳐맵을 사용하지 않기 때문에, 겹치는 anchor들이 더 많을 수 있습니다. 이를 위해 먼저 보편적인 방법인 non-maximum suppression(NMS)를 영상에 적용 했을 때 바운딩 박스의 변동이 심하다는 문제가 발생했습니다. (NMS는 동일한 클래스에 대해 가장 많이 높은 confidence score를 가지는 바운딩 박스와 일정한 기준값(threshold) 이상으로 겹치는 바운딩 박스는 삭제하는 방식으로 후처리 하는데(IoU 기준), 종국에는 하나의 바운딩 박스가 남겨 집니다.) 

이를 해결하기 위해서, BlazeFace는 겹치는 바운딩 박스에 대해 하나의 가장 확률이 높은 바운딩 박스를 남기는 것이 아닌, 바운딩 박스에의 regression 파라미터를 가중치로 이용한 가중평균을 구하여 바운딩 박스를 만들었습니다. 이를 적용한 실험에서 연산 상 추가적인 비용은 보이지 않았고, 10%의 accuracy 향상을 얻었다고 합니다. 

추가적으로 이미지에 약간의 이동을 취하여(jittering) 이동하기 전의 예측 값과 이동 후 예측 값의 MSE를 기준(regression의 품질 이라고 표현)으로 두 후처리의 결과를 비교 했을 때, 모바일의 전면 카메라의 경우 40% 감소, 후면 카메라에 대해서는 30%가 감소했다고 합니다. 

##### 실험 상세 

논문의 실험에서, 네트워크 학습에는 66000개의 이미지를 사용했고, 2000개의 이미지에 대해서 테스트를 했습니다. 

*BlazeFace* 는 전/후면 카메라 간의 차이를 고려해 각 각 따로 모델을 만들었고, 이미지에서 얼굴의 면적이 전면 카메라의 경우 20% 이상인 데이터 만 사용했다고 합니다. (후면 카메라는 5%) 그리고 regression 파라미터의 오차에 대해서는 Inter-ocular distance(IOD; 눈 과 눈 사이의 거리)를 이용해 정규화했고, 그 오차의 절댓값의 중앙값은 IOD의 7.4% 였다고 합니다. 그리고 이전 단락에서 언급한 jittering 지표는 IOD의 3% 였다고 합니다. 

다음 표는 여러 모바일 디바이스에서의 ms 단위의 Inference time 비교 결과를 보여줍니다. 결과를 보면, MobileNet-V2 기반의 SSD보다 성능과 속도 면에서 더 나은 결과를 내고 있습니다. 

![](https://www.dropbox.com/s/1gsc1y6edy8xp57/InferenceTimecomparison.png?raw=1)

##### 결론

BlazeFace가 이런 성능을 낼 수 있었던 요인은 다음 두 가지 라고 생각합니다. 

1.  GPU에서의 연산을 효율적으로 하기 위해 8x8 이하 크기의 피쳐맵을 사용하지 않음으로써, convolution layer를 절약한 것이 추론 시간에 큰 영향을 주었을 것이고, 그럼에도 5x5의 비교적 큰 커널 사이즈를 사용함으로써 그 감소 분에 의한 수용영역의 크기감소를 상쇄 할 수 있었습니다. 게다가 Depth-wise separable 연산에서는 커널 사이즈의 증가분에 대한 비용과 시간의 증가가 상대적으로 적었습니다.

2. 성능에 대해서는 같은 크기의 SSD 구조에서 보다 더 많은 바운딩 박스를 사용함으로써 더 나은 성능을 얻을 수 있었다고 생각했고, 예측 품질에 대해서(바운딩 박스 위치 예측의 변동성) 후처리 방법을 NMS 대신 tie resolution strategy 를 사용함으로써 박스의 위치를 변동이 적게 예측 할 수 있었을 것입니다. 