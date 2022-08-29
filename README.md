# 2022-SW-AI-competition
# Predict psychological tendency. 
### (2022.08.01 ~ 2022.08.26 Qualifying round)

## Outline
I did some experiments

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

----

