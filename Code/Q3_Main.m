%% Data processing
clear;
load('spamData.mat');
[num_train, num_feature] = size(Xtrain);
[num_test,~] = size(Xtest);
train_log = log(Xtrain+0.1);
test_log = log(Xtest+0.1);

%% Fit a logistic regression model
lambda = [1:10,15:5:100];
m = length(lambda);
train_ext = [ones(num_train,1),train_log];
test_ext = [ones(num_test,1),test_log];
for i = 1:m
	w0 = zeros(num_feature+1,1);
    while 1
        X = sigm(train_ext*w0);
        g = train_ext'*(X-ytrain) ...
            + lambda(i).*[0;w0(2:num_feature+1)];
        H = train_ext'*diag(X)*diag(1-X)*train_ext ...
            + lambda(i).*diag([0,ones(1,num_feature)]);
        w1 = w0 - H\g;
        if (w1-w0)'*(w1-w0) < 10^(-1)
            break
        end
        w0 = w1;
    end
    predict_train = sigm(train_ext*w1)>0.5;
    error_train(i) = sum(xor(predict_train,ytrain))/num_train;
    predict_test = sigm(test_ext*w1)>0.5;
    error_test(i) = sum(xor(predict_test,ytest))/num_test;
end

%% Plot and print the training and test error rates
figure(1)
plot(lambda,error_train*100)
hold on
plot(lambda,error_test*100)
title('Training and Test Error Rates vs \lambda')
xlabel('\lambda')
ylabel('Error Rates / %')
legend('Training','Test')
grid on;
fprintf('Training error rates for lambda= 1, 10, 100: %.2f%% %.2f%% %.2f%%\n', ...
    error_train([1,10,m])*100) 
fprintf('Test error rates for lambda= 1, 10, 100: %.2f%% %.2f%% %.2f%%\n', ...
    error_test([1,10,m])*100)

%% Function for sigm
function output = sigm(input)
    output = 1./(1+exp(-input));
end

