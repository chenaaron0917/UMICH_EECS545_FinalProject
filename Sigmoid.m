% EECS545 Final Project 32
% Haochen Wu, Yen-Yu Hsu, Shichao Zeng
% Online Sequential Extreme Learning Machine on Handwritten Digits Classification
% Sigmoidal activation function
function H = Sigmoid(X,W,b)
Z=X*W';
ind=ones(1,size(X,1));
B=b(ind,:);      
Z=Z+B;
H=1./(1+exp(-Z));
end