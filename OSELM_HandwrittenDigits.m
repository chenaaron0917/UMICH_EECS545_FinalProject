% EECS545 Final Project 32
% Haochen Wu, Yen-Yu Hsu, Shichao Zeng
% Online Sequential Extreme Learning Machine on Handwritten Digits Classification
% OS-ELM
clc;clear;close all
%% Data Processing
% Get Data from MNIST
[trainImages, trainLabels, dim] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[testImages, testLabels, dim] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
% View Image
% i=404;
% imagesc(reshape(trainImages(:,i),[dim dim]))
% title(num2str(trainLabels(i)))
% Set up
trainT=trainLabels; trainX=trainImages';
testT=testLabels; testX=testImages';
[nTrain,d]=size(trainX); 
nTest=size(testX,1);
nInputNeurons=d;
nClass=10; % 0,1,2,3,4,5,6,7,8,9
rng(0)
nOutputNeurons=nClass;
% Transfer Labels through One Hot Encoder
YTrain=one_hot_encode(trainT,nClass);
YTest=one_hot_encode(testT,nClass);

%% Boosting Phase/ Batch ELM
% Set N0=NTrain to do ELM in Batch Mode
tic
nHiddenNeurons=1024;
N0=round(nHiddenNeurons*1.5);
X0=trainX(1:N0,:); 
Y0=YTrain(1:N0,:);
W=rand(nHiddenNeurons,nInputNeurons)*2-1;
b=rand(1,nHiddenNeurons)*2-1;
H0=Sigmoid(X0,W,b);
M0=pinv(H0'*H0);
beta0=M0*H0'*Y0;
% Uncomment this section to see the error of ELM Batch Mode
%{
HTest=Sigmoid(testX,W,b);
Y=HTest * beta0;
[E,VAL,IDX,YY]=getError(Y,testT);
%}
%% Sequential Learning Phase
batch=64;
Hk=H0;Mk=M0;betak=beta0;
j=1;E=[];EE=[];
for k=N0:batch:nTrain
    if k+batch<=nTrain
        Xk = trainX(k+1:k+batch,:);    Yk=YTrain(k+1:k+batch,:);
    else % last batch might be less than batch size
        Xk = trainX(k+1:nTrain,:);    Yk=YTrain(k+1:nTrain,:);
        batch=size(Xk,1);
    end
    % Recursive Least Square
    Hk=Sigmoid(Xk,W,b);
    Mk=Mk-Mk*Hk'/(eye(batch)+Hk*Mk*Hk')*Hk*Mk; 
    betak=betak+Mk*Hk'*(Yk-Hk*betak);
    % Uncomment this section to plot Error as data arrives
    %{
    HTest=Sigmoid(testX,W,b);
    Y=HTest * betak;
    [E,VAL,IDX,YY]=getError(Y,testT);
    EE=[EE E];
    %}
end
%% Apply Trained Model to Test Data
HTest=Sigmoid(testX,W,b);
Y=HTest * betak;
[E,VAL,IDX,YY]=getError(Y,testT);
% Plot misclassified samples
figure(1)
for k=1:20
    subplot(4,5,k)
    imagesc(reshape(testX(IDX(k+60),:),[dim dim]))
    title(num2str(YY(IDX(k+60))))
end