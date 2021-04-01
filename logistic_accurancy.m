function [acc] = logistic_accurancy(labels,Y)
%Rounding the prediction value and comparing with the groundtruth for
%accracy calculation
Y=round(Y);
acc=(sum(labels==Y)/numel(labels))*100;
end

