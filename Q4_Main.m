%% Data processing
clear;
load('spamData.mat');
[num_train,~] = size(Xtrain);
[num_test,~] = size(Xtest);
train_log = log(Xtrain+0.1);
test_log = log(Xtest+0.1);

%% Implement a KNN classifier
K = [1:10,15:5:100];
m = length(K);
count_train = zeros(m,1);
count_test = zeros(m,1);
for i = 1:num_train
    index = Euclid(train_log(i,:),train_log);
	for j = 1:m
        n1 = sum(ytrain(index(1:K(j))));
        y = n1 > K(j)/2;
        count_train(j) = count_train(j) + xor(y,ytrain(i));         
    end
end
error_train = count_train/num_train;

for i = 1:num_test
	index = Euclid(test_log(i,:),train_log);
    for j = 1:m
        n1 = sum(ytrain(index(1:K(j))));
        y = n1 > K(j)/2;
        count_test(j) = count_test(j) + xor(y,ytest(i));
	end
end
error_test = count_test/num_test;

%% Plot and print the training and test error rates
figure(1)
plot(K,error_train*100)
hold on
plot(K,error_test*100)
title('Training and Test Error Rates vs K')
xlabel('K')
ylabel('Error Rates / %')
legend('Training','Test')
grid on;
fprintf('Training error rates for K= 1, 10, 100: %.2f%% %.2f%% %.2f%%\n', ...
    error_train([1,10,m])*100) 
fprintf('Test error rates for K= 1, 10, 100: %.2f%% %.2f%% %.2f%%\n', ...
    error_test([1,10,m])*100)

%% Function for Euclidean distance
function output = Euclid(input1,input2)
    [num_train, ~] = size(input2);
    dist = sqrt(sum(power(repmat(input1,num_train,1)-input2,2),2));
    [~, output] = sort(dist);
end