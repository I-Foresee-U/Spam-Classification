%% Data processing
clear;
load('spamData.mat');
[num_train, num_feature] = size(Xtrain);
[num_test, ~] = size(Xtest);
num_spam = [sum(ytrain), num_train-sum(ytrain)];
index = {find(ytrain==1), find(ytrain==0)};
train_bin = Binarize(Xtrain);
test_bin = Binarize(Xtest);

%% Statistics on the features
for i = 1:2
    feature{2*i-1} = sum(train_bin(index{i},:));
    feature{2*i} = num_spam(i)-feature{2*i-1};
end

%% Fit a Beta-Binomial naive Bayes classifier
alpha = 0:0.5:100;
lambda = sum(ytrain)/num_train;
for i = 1:length(alpha)
    pseudocounts = num_spam+2*alpha(i);
    for j = 1:2
        Beta_train{j} = log( (train_bin.*repmat(feature{2*j-1},num_train,1) ...
            + (1-train_bin).*repmat(feature{2*j},num_train,1) ...
            + alpha(i)) /pseudocounts(j) );
        Beta_test{j} = log( (test_bin.*repmat(feature{2*j-1},num_test,1) ...
            + (1-test_bin).*repmat(feature{2*j},num_test,1) ...
            + alpha(i)) /pseudocounts(j) );
    end
    predict_train = (sum(Beta_train{1},2)+log(lambda)) ...
        > (sum(Beta_train{2},2)+log(1-lambda));
    error_train(i) = sum(xor(predict_train,ytrain)) / num_train;
    predict_test = (sum(Beta_test{1},2)+log(lambda)) ...
        > (sum(Beta_test{2},2)+log(1-lambda));
    error_test(i) = sum(xor(predict_test,ytest)) / num_test;
end

%% Plot and print the training and test error rates
figure(1)
plot(alpha,error_train*100)
hold on
plot(alpha,error_test*100)
title('Training and Test Error Rates vs \alpha')
xlabel('\alpha')
ylabel('Error Rates / %')
legend('Training','Test')
grid on;
fprintf('Training error rates for alpha= 1, 10, 100: %.2f%% %.2f%% %.2f%%\n', ...
    error_train([3,21,201])*100) 
fprintf('Test error rates for alpha= 1, 10, 100: %.2f%% %.2f%% %.2f%%\n', ...
    error_test([3,21,201])*100) 

%% Function for binarization
function output = Binarize(input)
    index = find(input>0);
    input(index) = 1;
    output = input;
end