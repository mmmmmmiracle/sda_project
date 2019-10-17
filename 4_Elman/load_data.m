train_file = 'train_2015_59_24.csv';
test_file = 'test_2015_59_24.csv';

data_train = importdata(train_file);
data_train = data_train';
[len,temp] = size(data_train);
x_train = data_train(1:len-1,:);
y_train = data_train(len , :);

data_test = importdata(test_file);
data_test = data_test';
[len,temp] = size(data_train);
x_test = data_test(1:len-1,:);
y_test = data_test(len ,: );