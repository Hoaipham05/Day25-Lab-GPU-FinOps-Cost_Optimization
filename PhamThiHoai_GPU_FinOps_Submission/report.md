# Báo cáo GPU FinOps & Cost Optimization

## Thông tin sinh viên

- Họ và tên: Phạm Thị Hoài
- MSSV: 2A202600269

## 1. Giới thiệu

Bài lab này mô phỏng một chu trình GPU FinOps hoàn chỉnh, từ quan sát tài nguyên GPU, ghi nhận chi phí, tối ưu bằng spot instance và autoscaling, cho đến đánh giá chi phí huấn luyện trên GPU thật. Ngoài phần mock platform chạy qua Docker Compose, notebook còn thực hiện huấn luyện ResNet-18 trên CIFAR-10 để so sánh chế độ FP32 và Mixed Precision (AMP).

Mục tiêu chính của bài làm:

- Hiểu trạng thái cluster GPU và các chỉ số tài nguyên.
- Theo dõi chi phí workload và mức tiết kiệm.
- Đánh giá spot instance, autoscaling, waste và khuyến nghị tối ưu.
- So sánh hiệu quả chi phí trên GPU thật.
- Xây dựng phân tích nâng cao về scaling, forecasting và roadmap tối ưu.

## 2. Phân tích từng phần

### Part 1-7: Mock GPU FinOps Platform

#### 2.1. GPU Cluster Monitoring

Cluster ban đầu có 4 node với tổng cộng 8 GPU. Ở thời điểm bắt đầu, tất cả GPU đều đang idle, average utilization chỉ khoảng 5.9%, memory đang sử dụng 10.9 GB trên tổng 288 GB và tổng công suất khoảng 314 W.

Kết quả này cho thấy hệ thống đang ở trạng thái dự phòng tài nguyên, có khả năng tiếp nhận workload mới nhưng cũng tồn tại khả năng lãng phí nếu trạng thái idle kéo dài.

#### 2.2. Workload Submission và Billing

Notebook đã submit 4 workload:

- `train-resnet-001`
- `train-bert-002`
- `inference-api-003`
- `train-llm-004`

Sau khi submit, có 5/8 GPU bận, utilization tăng lên 47.1%. Billing summary ghi nhận:

- Tổng chi phí: `$1.1949`
- Tổng savings: `$1.2927`
- Budget utilization: `1.2%`
- Alert status: `OK`

Kết quả này cho thấy hệ thống FinOps không chỉ ghi lại chi phí tuyệt đối mà còn giúp tách biệt hiệu quả tiết kiệm từ workload spot.

#### 2.3. Spot Instance Management

Giá spot ở thời điểm chạy có mức discount khá tốt trên các dòng GPU. Sau khi request spot instance, cả 3 instance mẫu đều được cấp phát thành công. Trong bài test preemption, có 1 instance bị reclaim, còn 2 instance tiếp tục chạy.

Báo cáo savings cho thấy:

- Spot cost: `$0.0001`
- On-demand equivalent: `$0.0002`
- Savings: `17.2%`

Phần này minh họa rõ trade-off của spot instance: chi phí giảm, nhưng có rủi ro preemption nên phù hợp với workload có thể retry hoặc checkpoint.

#### 2.4. Autoscaling

Autoscaling policy được hiển thị và cập nhật thành công. Trong 5 chu kỳ evaluation, utilization 47.1% nằm trong ngưỡng an toàn nên autoscaler chọn `NO_ACTION`.

Điều này hợp lý vì:

- Scale up lúc này sẽ làm tăng chi phí không cần thiết.
- Scale down lúc này có thể làm giảm khả năng xử lý workload đang chạy.

Autoscaling giúp duy trì cân bằng giữa performance và cost, tránh over-provisioning.

#### 2.5. Cost Snapshot, Waste và Recommendations

Notebook ghi 5 cost snapshot liên tiếp, mỗi snapshot có:

- Total cost per interval: `$0.038056`
- Idle cost: `$0.008833`
- Waste: `23.2%`

Waste report tổng hợp:

- Average waste: `23.2%`
- Total idle cost: `$0.044165`
- Potential monthly savings: `$2289.51`
- Severity: `LOW`

Hệ thống cũng đưa ra hai khuyến nghị chính:

- Sử dụng spot instance cho workload phù hợp.
- Sắp lịch job không gấp trong khung giờ chi phí thấp hơn.

#### 2.6. Dashboard và Full Workflow

Dashboard tổng hợp cho thấy:

- 8 GPU trên 4 node
- Utilization: `47.1%`
- Billing: `$1.1949 / $100`
- Spot savings: `$0.0030`
- Waste: `23.2%`

Trong workflow tổng hợp, sau khi thêm workload nặng:

- Utilization tăng lên `75.8%`
- Autoscaler ra quyết định `scale_up`
- Cost interval: `$0.040000`
- Waste giảm xuống `4.9%`
- Final billing: `$1.3640`
- Total savings: `$1.4151`

Chu trình này thể hiện ý tưởng FinOps quan trọng: tối ưu tài nguyên không có nghĩa là cắt giảm tối đa, mà là tăng hiệu quả chi phí trong khi vẫn đáp ứng nhu cầu tải.

### Part 8: Real GPU Workload trên Kaggle

#### 2.7. Môi trường GPU thật

Notebook detect được:

- GPU: `Tesla T4`
- Memory: `15.6 GB`
- Giá tham chiếu: `$0.35/hr`
- CUDA: `12.8`
- `pynvml`: available

Diagnostic thu thập được telemetry thật như GPU utilization, memory, power và temperature.

#### 2.8. FP32 Training

Kết quả huấn luyện FP32 trong 3 epoch:

- Tổng thời gian: `123.4s`
- Peak memory: `0.82 GB`
- Avg GPU utilization: `95.4%`
- Avg power: `65.9W`
- Avg temperature: `66.4C`
- Estimated cost: `$0.011994`

FP32 đóng vai trò baseline đầy đủ để so sánh hiệu quả tối ưu.

#### 2.9. AMP Training

Chế độ AMP rút ngắn đáng kể thời gian train. Phần tổng hợp so sánh cho thấy:

- FP32 time: `123.4s`
- AMP time: `59.0s`
- Tốc độ nhanh hơn: `2.09x`
- Peak memory giảm: `0.22 GB`
- Cost saving: `$0.006259`
- Cost saving percentage: `52.2%`

Kết quả này cho thấy Mixed Precision là một trong những kỹ thuật tối ưu FinOps rất hiệu quả khi model và GPU hỗ trợ.

#### 2.10. Báo cáo cost về gateway

Notebook đã gửi cost thật về FinOps gateway:

- FP32 workload cost: `$0.012000`
- AMP workload cost: `$0.001700`
- Savings từ AMP workload: `$0.004000`
- Project `real-gpu-lab` có tổng cost `$0.013700`

Điều này mở rộng bài lab từ mô phỏng sang một mô hình gần với thực tế hơn: chi phí của job GPU thật được nhập lại vào hệ thống tổng hợp FinOps.

### Part 8.5: Advanced GPU Cost Optimization

#### 2.11. Multi-GPU Cost Analysis

Phân tích trên GPU A100 cho thấy:

- Phương án chi phí thấp nhất: `1 GPU`
- Phương án cost/performance tốt nhất: `8 GPU`

Với 8 GPU:

- Speedup factor: `5.8`
- Runtime: `0.345h`
- Total cost: `$10.12`
- Cost per speedup unit: `$1.75`

Nhận xét quan trọng là hệ thống không nên chỉ tối ưu theo một chiều. Nếu mục tiêu là minimum spend, 1 GPU tốt hơn; nếu mục tiêu là thời gian hoàn thành và hiệu quả hiệu năng, cấu hình nhiều GPU có thể đáng giá hơn.

#### 2.12. Project Cost Forecasting

Forecast tổng hợp cho dự án mẫu:

- Total base cost: `$3551.20`
- Contingency: `$710.24`
- Expected with contingency: `$4261.44`
- Confidence interval: `$1645.34 - $5457.06`

Forecast này cho thấy chi phí dự án GPU cần được lập kế hoạch theo phase và kèm buffer rủi ro, thay vì chỉ nhìn giá GPU mỗi giờ.

#### 2.13. Optimization Opportunity Analysis

Baseline cost của cấu hình phân tích:

- `$1468.00`

Sau khi xếp ưu tiên các biện pháp tối ưu, hệ thống ước tính:

- Cumulative savings: `$1288.32`
- Final cost after all optimizations: `$179.68`

Kết quả này minh họa giá trị của việc gom nhiều biện pháp nhỏ thành một roadmap tối ưu tổng thể, thay vì chỉ phụ thuộc vào một biện pháp riêng lẻ.

#### 2.14. Challenge Strategy

Challenge scenario là fine-tuning LLM:

- Baseline: `8x A100` trong `200h`
- Baseline cost: `$5872.00`

Chiến lược đề xuất:

- Giữ cấu hình `8x A100` để đảm bảo deadline.
- Dùng AMP để giảm cost và thời gian.
- Kết hợp batch-size tuning và early stopping.
- Loại bỏ giải pháp spot nếu rủi ro preemption vượt mức chấp nhận.

Kết quả ước tính:

- Optimized training cost: `$294.56`
- Strategy forecast đã kèm validation và contingency.

Hướng tiếp cận đúng không chỉ dựa trên savings, mà còn phải cân bằng budget, accuracy guardrail và risk constraint.

## 3. Kết luận và bài học rút ra

Qua bài lab này, có thể rút ra một số bài học chính:

- GPU FinOps cần kết hợp monitoring, billing, forecasting và optimization thay vì chỉ theo dõi tổng chi phí.
- Spot instance giúp tiết kiệm chi phí đáng kể nhưng cần đánh giá preemption risk.
- Autoscaling giúp giảm waste khi workload biến động.
- Mixed Precision là kỹ thuật có tác động lớn đến time-to-train và cost trên GPU thật.
- Khi lập kế hoạch dự án GPU, cần có forecast, contingency và roadmap ưu tiên tối ưu.

Tổng thể, bài lab cho thấy một quy trình FinOps tốt có thể giảm lãng phí, cải thiện hiệu quả tài nguyên và giữ chi phí dự án trong ngưỡng kiểm soát.
