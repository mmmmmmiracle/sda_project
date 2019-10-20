import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def load_data(file_name,start_point):
    source_df = pd.read_csv(file_name,index_col=0)   #读入csv文件
    index_col = pd.date_range(start_point,periods=len(source_df),freq='h')   #产生日期index
    source_df = pd.DataFrame(np.array(source_df),index=index_col,columns=source_df.columns) #产生标准日期格式的DataFrame
    return source_df

def show_distribute(source_df):
    source_df[['zhexi_in','xiaoxi_out']].hist(bins=4)  #上下游进出水量分布情况
    plt.show()
    source_df[['lengshuijiang_add','xinhua_add','zhexi_add']].hist(bins=4) #降雨信息分布情况
    plt.show()

def handle_outlier_points(source_df):
    #replace the negetive number in column 'zhexi_in' with zero
    print('zhexi in negtive points num:',source_df[source_df['zhexi_in'] < 0]['zhexi_in'].count())
    # source_df.ix[source_df['zhexi_in'] < 0,'zhexi_in'] = 0
    source_df['zhexi_in'] = source_df['zhexi_in'].apply(lambda x:x if(x>0) else 0)
    print('zhexi_in negtive points num:',source_df[source_df['zhexi_in'] < 0]['zhexi_in'].count())

    print('xiaoxi out negtive points num:',source_df[source_df['xiaoxi_out'] < 0]['xiaoxi_out'].count())
    # source_df.ix[source_df['xiaoxi_out'] < 0,'xiaoxi_out'] = 0
    source_df['xiaoxi_out'] = source_df['xiaoxi_out'].apply(lambda x:x if(x>0) else 0)
    print('xiaoxi out negtive points num:',source_df[source_df['xiaoxi_out'] < 0]['xiaoxi_out'].count())
    return source_df

def handle_nan(source_df):
    source_df.fillna(axis=0,method='ffill')  #纵向用缺失值前面的值替换缺失值"
    print(source_df.zhexi_in.notnull().value_counts())
    print(source_df.xiaoxi_out.notnull().value_counts())
    print(source_df.lengshuijiang_add.notnull().value_counts())
    print(source_df.xinhua_add.notnull().value_counts())
    print(source_df.zhexi_add.notnull().value_counts())
    return source_df

# a.最近24小时的水量平均值特征  
def water_avg(source_df):
    source_df['zhexi_in_day_avg'] = source_df[['zhexi_in']].rolling(window=24).mean()
    source_df['xiaoxi_out_day_avg'] = source_df[['xiaoxi_out']].rolling(window=24).mean()
    return source_df

# b.最近24小时的降雨量总和特征，以及降雨强度类别特征 
def rainfall_total(source_df):
    source_df['lengshuijiang_add_day_rainfall_total'] = source_df[['lengshuijiang_add']].rolling(window=24).sum()
    source_df['xinhua_add_add_day_rainfall_total'] = source_df[['xinhua_add']].rolling(window=24).sum()
    source_df['zhexi_add_day_rainfall_total'] = source_df[['zhexi_add']].rolling(window=24).sum()
    source_df = source_df.dropna()
    return source_df

def rainfall_level_determination(rainfall):
    if rainfall < 0.0:
        return -1;
    elif rainfall < 10.0:  #小雨
        return 'level1';
    elif rainfall < 25.0:  #中雨
        return 'level2';
    elif rainfall < 50.0:  #大雨
        return 'level3';
    elif rainfall < 100.0: #暴雨
        return 'level4';
    elif rainfall < 250.0: #大暴雨
        return 'level5';
    else:                  #特大暴雨
        return 'level6';

def rainfall_heavy(source_df):
    for record_index in list(source_df.index):
        source_df.ix[record_index,'lengshuijiang_add_level'] = rainfall_level_determination(
            source_df.ix[record_index,'lengshuijiang_add_day_rainfall_total'])
        source_df.ix[record_index,'xinhua_add_level'] = rainfall_level_determination(
            source_df.ix[record_index,'xinhua_add_add_day_rainfall_total'])
        source_df.ix[record_index,'zhexi_add_level'] = rainfall_level_determination(
            source_df.ix[record_index,'zhexi_add_day_rainfall_total'])
    source_df = pd.get_dummies(source_df,columns=['lengshuijiang_add_level','xinhua_add_level','zhexi_add_level'],dtype=np.float32)
    for level in range(1,7):
        for prefix in ['lengshuijiang_add_level_level','xinhua_add_level_level','zhexi_add_level_level']:
            tmp_feature_name = f'{prefix}{level}'
            # print(tmp_feature_name not in source_df.columns)
            if tmp_feature_name not in source_df.columns:
                source_df[tmp_feature_name] = 0.0
    return source_df


# c.最近period个进水量值、降雨量值  
# d.上游period个时刻的出水量，经过offset个小时的延迟，影响到当前时刻

columns = ['xiaoxi_out','xiaoxi_out_day_avg','zhexi_in','zhexi_in_day_avg']

def create_shift_features(data,shift_columns=columns,offsets=[0,0,48,48],periods=[12,12,12,12]):
    for i in range(1,len(shift_columns)+1):
        column,offset,period = shift_columns[i-1],offsets[i-1],periods[i-1]
        for j in range(1,period+1):
            data[f'{column}_shift{offset}_{j}'] = data[column].shift((offset+j))
    return data.dropna()


def whole_flow(source_df):
    # print(source_df.shape)
    source_df = handle_outlier_points(source_df)
    # print(source_df.shape)
    source_df = handle_nan(source_df)
    # print(source_df.shape)
    source_df = water_avg(source_df)
    # print(source_df.shape)
    source_df = rainfall_total(source_df)
    # print(source_df.shape)
    source_df = rainfall_heavy(source_df)
    # print(source_df.shape)
    return source_df


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5),scoring='mean_squared_error',n_jobs=16):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """
    
    from sklearn.model_selection  import learning_curve

    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()


def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    
    返回:
    mape -- MAPE 评价指标
    """
    
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape

def mse(y_true,y_pred):
    return mean_squared_error(y_true,y_pred)