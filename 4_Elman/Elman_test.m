function [predict,mse_test]=Elman_test(x,y)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
load water_net1.mat;

predict = sim(net,x);
error = predict - y;
mse_test = mse(error);
fprintf('error= %f\n', error);
plot(1:length(y),y,'b-',1:length(y),predict,'r-');
title(['使用测试集数据测试   mse值为',num2str(mse_test)]);
legend('真实值','测试结果');
end

