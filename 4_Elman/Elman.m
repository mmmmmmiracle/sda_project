function [predict_train , mse_train, predict_test, mse_test] = Elman(x_train,y_train,x_test,y_test)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明

% 设置迭代次数
net=elmannet;
net.trainParam.epochs=2000;
% 初始化
net=init(net); 
net=train(net,x_train,y_train,'useGPU','only');
save water_net net

%% 使用训练数据测试一次
predict_train=sim(net,x_train);
error=predict_train-y_train;
mse_train=mse(error);
fprintf('error= %f\n', error);
subplot(2,1,1);
plot(1:length(y_train),y_train,'b-',1:length(y_train),predict_train,'r-');
title(['使用训练集数据测试   mse值为',num2str(mse_train)]);
legend('真实值','测试结果');

%% 使用测试数据测试
predict_test=sim(net,x_test);
error=predict_test-y_test;
mse_test=mse(error);
fprintf('error= %f\n', error);
subplot(2,1,2);
plot(1:length(y_test),y_test,'b-',1:length(y_test),predict_test,'r-');
title(['使用测试集数据测试   mse值为',num2str(mse_test)]);
legend('真实值','测试结果');
end

