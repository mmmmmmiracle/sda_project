test_file = 'season_test/2016_spring.csv';

data_test = importdata(test_file);
data_test = data_test';
[len,temp] = size(data_test);
x_test = data_test(1:len-1,:);
y_test = data_test(len,:);
