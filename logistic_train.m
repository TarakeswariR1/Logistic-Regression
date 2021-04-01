function [weight] = logistic_train(data, labels, epsilon, maxiter)
%Implemented a Newton-Raphson (IRLS) iterative 
weight=zeros(size(data, 2), 1);
update=0;
epsilon= 1e-5;
maxiter=1000;
halt = Inf;
%To calculate the weight 

while ((update <= maxiter) && halt>epsilon)
    h=sigmoid(data*weight);
    I=eye(length(data));
    R1=diag(h .*(1-h));
    R=R1+I;
    z=(data * weight) - (R^(-1)*(h-labels));
    part1=(data' * R * data)^(-1);
    part2=data' * R * z;
    %An iterative process of weight update
    weight=part1*part2 ;
    
    %using the new values of weights to calculate the prediction
    y=sigmoid(data * weight);
    abs_diff= abs(y-h);
    halt=mean(abs_diff);
    update=update+1;
    

    
end
end