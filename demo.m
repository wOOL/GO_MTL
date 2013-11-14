%% Synthetic Data
close all;clear;
rng(1943);

load toy;
k = 3;
Maxsteps=100;
lambda=0.1;
mu = 5;
O = 'R';
[L,S,W] = GO_MTL(X_train,Y_train,k,Maxsteps,lambda,mu,O);
Y = cell2mat(Y_test);
T = size(X_test,1);
Y_hat = cell(T,1);
for t = 1:T
    Y_hat{t} = X_test{t}*W(:,t);
end
Y_hat = cell2mat(Y_hat);
RMSE = sqrt(mean((Y - Y_hat).^2));
fprintf(sprintf('Independent Training RMSE: %f\n',RMSE));
W = L*S;
Y_hat = cell(T,1);
for t = 1:T
    Y_hat{t} = X_test{t}*W(:,t);
end
Y_hat = cell2mat(Y_hat);
RMSE = sqrt(mean((Y - Y_hat).^2));
fprintf(sprintf('GOMTL RMSE: %f\n',RMSE));
colormap(gray);
imagesc(abs(S));
colorbar;
%% MNITS Data
close all;clear;
rng(1943);

load mnistPCA_1k;
k = 10;
Maxsteps=100;
lambda=0.1;
mu = 5;
O = 'C';
[L,S,W] = GO_MTL(X_train,Y_train,k,Maxsteps,lambda,mu,O);
Y = cell2mat(Y_test);
T = size(X_test,1);
Y_hat = cell(T,1);
for t = 1:T
    Y_hat{t} = sign(1./(1+exp(-X_test{t}*W(:,t)))-0.5);
end
Y_hat = cell2mat(Y_hat);
Err = mean(Y~=Y_hat);
fprintf(sprintf('Independent Training Error Rate: %f\n',Err));
W = L*S;
Y_hat = cell(T,1);
for t = 1:T
    Y_hat{t} = sign(1./(1+exp(-X_test{t}*W(:,t)))-0.5);
end
Y_hat = cell2mat(Y_hat);
Err = mean(Y~=Y_hat);
fprintf(sprintf('GOMTL Error Rate: %f\n',Err));
colormap(gray);
imagesc(abs(S));
colorbar;