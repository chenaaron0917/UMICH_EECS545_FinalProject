% EECS545 Final Project 32
% Haochen Wu, Yen-Yu Hsu, Shichao Zeng
% Online Sequential Extreme Learning Machine on Handwritten Digits Classification
% Get misclassification rate
% returns E=misclassification rate
% VAL=misclassified value
% IDX=misclassified index
% Y=misclassified class
function [E,VAL,IDX,Y]=getError(Ypred,T)
error=0;
VAL=[];
IDX=[];
Y=T;
for i=1:size(Ypred,1)
    % max value in Yi is the identified class
    [val, idx] = max(Ypred(i,:));
    Y(i)=idx-1;
    if T(i)~=Y(i)
        error=error+1;
        VAL=[VAL val];
        IDX=[IDX i];
    end
end
% error
E=error/size(Ypred,1);
end