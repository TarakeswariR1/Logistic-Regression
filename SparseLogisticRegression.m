Alzimer_data=load("/Users/tarakeswariramachandra/Documents/Michigan State University/CSE 847/Logistic REgression/ad_data.mat");
Alzimer_features=load("/Users/tarakeswariramachandra/Documents/Michigan State University/CSE 847/Logistic REgression/feature_name.mat");

train_data = Alzimer_data.X_train;
test_data = Alzimer_data.X_test;
train_label = Alzimer_data.y_train;
test_label = Alzimer_data.y_test;

% Setup other L1 regularization parameters
par  = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
count=numel(par);
Number_of_features= zeros(size(par));
AUC = zeros(size(par));
for i=1:count
    [w, c]=logistic_l1_train(train_data, train_label, par(i));
    Number_of_features(i)=sum(w~=0);
    test_y=test_data*w;
    %acc=sparse_accuracy(test_label, test_y);
    cp = classperf(test_label>=0,test_y>=0);
    acc= cp.CorrectRate;
    %c1= confusionmat(test_label,test_y);
    [X, Y, T, AUC(i)]=perfcurve(test_label, test_y, 1);
    figure;
    plot(X, Y, '-o');
    title('{\bf False positive and True Positive}', num2str(par(i)));
    xlabel('False Positive');
    ylabel('True Positive');
    fprintf("Regularization term :%f, Number of non-zero features :%f, Accuracy:%f ,AUC :%f\n", par(i), Number_of_features(i),acc, AUC(i));
end



figure;
plot(par, AUC, '-o');
title('{\bf Area under the curve}');
xlabel('L1 Regularization Parameter');
ylabel('AUC');

figure;
plot(par, Number_of_features, '*--');
title('{\bf Number of non-zero entries}');
xlabel('\bfL1 Regularization Parameter');
ylabel('Number of features');
ylim([0 200]);

