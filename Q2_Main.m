%% Data processing
clear;
load('spamData.mat');
[num_train, ~] = size(Xtrain);
[num_test, ~] = size(Xtest);
index = {find(ytrain==1), find(ytrain==0)};
train_log = log(Xtrain+0.1);
test_log = log(Xtest+0.1);

mean_train = {mean(train_log(index{1},:)),mean(train_log(index{2},:))};
var_train = {var(train_log(index{1},:)),var(train_log(index{2},:))};

%% Fit a Gaussian naive Bayes classifier
lambda = sum(ytrain)/num_train;
for i = 1:num_train
    Gau_train1 = -(train_log(i,:)-mean_train{1}).^2 ...
        ./(2*var_train{1}) -log(sqrt(2*pi*var_train{1}));
    Gau_train2 = -(train_log(i,:)-mean_train{2}).^2 ...
        ./(2*var_train{2}) -log(sqrt(2*pi*var_train{2}));
    predict_train(i) = (sum(Gau_train1)+log(lambda)) > (sum(Gau_train2)+log(1-lambda));
end
error_train = sum(xor(predict_train',ytrain));

for i = 1:num_test
    Gau_test1 = -(test_log(i,:)-mean_train{1}).^2 ...
        ./(2*var_train{1}) -log(sqrt(2*pi*var_train{1}));
    Gau_test2 = -(test_log(i,:)-mean_train{2}).^2 ...
        ./(2*var_train{2}) -log(sqrt(2*pi*var_train{2}));
    predict_test(i) = (sum(Gau_test1)+log(lambda)) > (sum(Gau_test2)+log(1-lambda));
end
error_test = sum(xor(predict_test',ytest));

%% Print the training and test error rates
fprintf('Training error rates for the log-transformed data: %.2f%%\n', ...
    error_train/num_train*100)
fprintf('Testing error rates for the log-transformed data: %.2f%%\n', ...
    error_test/num_test*100) 
