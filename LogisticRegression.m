Data=load("/Users/tarakeswariramachandra/Documents/Michigan State University/CSE 847/data.txt");
label=load("/Users/tarakeswariramachandra/Documents/Michigan State University/CSE 847/label.txt");
%Adding a bias term 
Data = [ones(size(Data,1),1),Data];
%Splitting the data into training and test dataset
test_data=Data(2001:4601,:);
train_data=Data(1:2000,:);
train_label=label(1:2000, :);
test_label=label(2001:4601, :);
%Below is the vector of training size, where we observe how accuracy
%changes with more training data size
train_size=[200, 500, 800, 1000, 1500, 2000];
count=numel(train_size);
%Vector to store all the accuracy 
acc=zeros(count);
for i =1:count
    n=train_size(i);
    data=train_data(1:n, :);
    labels=train_label(1:n, :);
    weight=logistic_train(data, labels);
    test_y=sigmoid(test_data*weight);
    acc(i)=logistic_accurancy(test_label, test_y);
    fprintf("Training size:%f, Accuracy:%f\n", n, acc(i));
end

figure;
plot(train_size, acc,'*--');
xlim([200 2000]);
ylim([80 100]);
title('{\bf Logistic Regression accuracy with varing training size}');
xlabel("Training size");
ylabel("Accuracy of prediction");
