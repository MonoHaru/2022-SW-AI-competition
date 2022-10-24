# 2022-SW-AI-competition
# Korean scene text recognition(OCR).
https://dacon.io/competitions/official/235970/overview/description 
### (2022.09.08 ~ 2022.10.07 Final round)

## Outline
[Public: 0.73909 #22, **Private: 0.73102 #23** :grin:]<br>
vitstr_base_patch16_224 is the best model of our model.<br>
ViTSTR ***https://github.com/roatienza/deep-text-recognition-benchmark***

#### Data experiments
1. Base train data
2. Train data with extra data which is added shuffled korean font print and handwrite data
3. Large extra with above data and augmented font print data   

#### Model experiments
1. Clova deep-text-recognition-benchmark model.
    https://github.com/clovaai/deep-text-recognition-benchmark
    1. TRBC(TPS-ResNet-BiLSTM-CTC)
    2. TRBA(TPS-ResNet-BiLSTM-Attn) Best model in that series.
    3. TVBA(TPS-VGG-BiLSTM-Attn)
    4. TUBA(TPS-Unet-BiLSTM-Attn)
    4. TEBA(TPS-Efficient-BiLSTM-Attn)
    5. TVBA(TPS-ViT-BiLSTM-Attn)
2. **ViTSTR** (, the best model of mine!) https://github.com/roatienza/deep-text-recognition-benchmark
3. PARSeq https://github.com/baudm/parseq
4. ~~TrOCR https://github.com/microsoft/unilm/tree/master/trocr~~

## Summary
<details>
<summary>Our journey</summary>
<div>
https://www.notion.so/AI-e217527b665149f7a3447d8af037ef0a
</div>
</details>
<br>

#### Data experiments result
1. Our models, on average,has reached about 60% accuracy when using this data.
2. Processed data except natural and augmentated data has improved to the accuracy about 70%. 
3. Overfitting has occurred... (has approximately reached 20% accuracy...)

#### Model experiments result
1. Most of this type of models reached above 70% accuracy. But deep models such as TVBA and TEBA didn't approach similar accuracy to other models. 
2. ViTSTR models(or TPS-ViTSTR) obtained to similar accuracy as above models. However these models gained higher accuracy than Clova models!
3. PARSeq model only focuses on English, not on other languages...(Not good for us)
4. ~~We don't have much time to try this method.~~


## Trubleshooting
1. Grayscale is a much better format than RGB.
2. timm(Pytorch Image Models) structure has been changed since version 0.5.4 and later.
3. When you want to classify sentences including white-space, **you must contain white-space in your character set!**

## To make greater model.
- All our models have been improved. So if you use our pre-trained models, you can reach more high accuracy.(We didn't have time to finish training models which are in pre-overfitting conditions by the end of this competition....)
- Edit our models for Korean characteristic. For instance we can find out that Korean has three parts in this letter. So we are possible to change to sementic segmentation problem.
- Using extra data similar to test data will help our models attain improved accuracy.

---
<br>

# Predict psychological tendency. 
https://dacon.io/competitions/official/235902/overview/description
### (2022.08.01 ~ 2022.08.26 Qualifying round)

## Outline
[Public: 0.88289 #59, **Private: 0.87972 #57** :bowtie:]<br>
Test3 in ipynb(submission_inverse_deep1_conv1d_pred_fold7_1.csv)<br>
I did some experiments using ***Neural network***

#### Data experiments
1. RobustScaler and MinMaxScaler
2. Use PCA.

#### Model experiments
1. Simple fc(Fully connected layer) with conv1d and batchNorm1d. - It's working!
2. deeper fc layer.
3. SEBlock added after fc.
4. Remove SEBlock and add dropout in each fc block.
5. Model architecture like U-Net or inverse

## Summary

#### Data experiments
1. There is no performance improvement in RobustScaler and MinMaxScaler(Performance was simillar).
2. PCA was same as above..(Didn't make remarkable performance.)

#### Model experiments
1. Conv1d method is working to reach better performance. 
- While I don't check private AUC scores at competition page, in my accuracy score, Our model can show only up to 76%...
2. It looked like to get better model but it was not....(simillar original's score with deeper's score) 
3. Same as 2.
4. It prevented overfitting however It eventually became overfitting model.
5. It was not working...(score was below 50%....;) )

## To make greater model.
- Neural network model requires a large amount of data!


